def gaussian_kl_divergence(mu1, sigma1, mu2, sigma2):
    """
    Compute KL divergence between two multivariate Gaussian distributions.
    Assumes diagonal covariance matrices.
    """
    var1 = sigma1.pow(2)
    var2 = sigma2.pow(2)
    kl = (var1 / var2 + (mu2 - mu1).pow(2) / var2 - 1 + var2.log() - var1.log()).sum(-1) * 0.5
    return kl.mean()
def DPO(sac_agent, prev_model, expert_states, expert_actions, greedy=False,steps=100,beta=0.1,random=False,clip_grad=False):
    assert expert_states.shape[0] == expert_actions.shape[0]
    prev_model.eval()
    batch_size = 256
    total_loss = 0
    total_margin = 0
    total_positive_reward = 0
    total_negative_reward = 0
    LOG_STD_MIN = -20
    LOG_STD_MAX = 2
    epsilon = 1e-3

    for i in range(steps):
            # state =
            # chosen_act =
            if random:
                 reject_act = torch.FloatTensor(np.random.uniform(-1,1,chosen_act.shape)).to(sac_agent.device)
            else:
                with torch.no_grad():

                    reject_act, reference_rejected_logps = prev_model(state, deterministic=False, with_logprob=True)
            # Clamp the reject action to the action space
            chosen_act = torch.clamp(chosen_act, -1+epsilon, 1-epsilon)
            reject_act = torch.clamp(reject_act, -1+epsilon, 1-epsilon)

            # Calculate log probabilities for chosen actions
            policy_chosen_logps = sac_agent.ac.pi.log_prob(state, chosen_act)
            policy_rejected_logps = sac_agent.ac.pi.log_prob(state, reject_act)

            with torch.no_grad():


                reference_chosen_logps = prev_model.log_prob(state, chosen_act)
                reference_rejected_logps = prev_model.log_prob(state, reject_act)


            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            reject_logratios = policy_rejected_logps - reference_rejected_logps
            
            positive_reward = chosen_logratios.detach().mean().item()
            negative_reward = reject_logratios.detach().mean().item()
            margin = positive_reward - negative_reward
            
            total_positive_reward += positive_reward
            total_negative_reward += negative_reward
            total_margin += margin


            logits = policy_chosen_logps - policy_rejected_logps - reference_chosen_logps + reference_rejected_logps

            # reverse kl(original DPO)
            losses = -torch.nn.functional.logsigmoid(beta * logits)

            loss = losses.mean()
            total_loss += loss.sum()

            sac_agent.pi_optimizer.zero_grad()
            loss.backward()
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(sac_agent.ac.pi.parameters(), max_norm=1.0)
            sac_agent.pi_optimizer.step()

    return total_loss, total_margin, total_positive_reward, total_negative_reward

def KTO(sac_agent, prev_model, expert_states, expert_actions,greedy=False, steps=100, beta=0.1, random=False, clip_grad=False):
    assert expert_states.shape[0] == expert_actions.shape[0]
    prev_model.eval()
    batch_size = 256
    total_loss = 0
    total_margin = 0
    total_positive_reward = 0
    total_negative_reward = 0
    desirable_weight = 1.0  # You can adjust this weight
    undesirable_weight = 1.0  # You can adjust this weight
    LOG_STD_MIN = -20
    LOG_STD_MAX = 2
    epsilon = 1e-3
    for i in range(steps):
            # state =
            # chosen_act =
            # reject_act =
            if random:
                reject_act = torch.FloatTensor(np.random.uniform(-1, 1, chosen_act.shape)).to(sac_agent.device)
            else:
                with torch.no_grad():
                    reject_act, reference_rejected_logps = prev_model(reject_state, deterministic=False, with_logprob=True)


            
            # Clamp the reject action to the action space
            reject_act = torch.clamp(reject_act, -1+epsilon, 1-epsilon)
            chosen_act = torch.clamp(chosen_act, -1+epsilon, 1-epsilon)
            # Calculate log probabilities for chosen actions
            policy_chosen_logps = sac_agent.ac.pi.log_prob(state, chosen_act)
            policy_rejected_logps = sac_agent.ac.pi.log_prob(state, reject_act)

            # Calculate KL divergence
            net_out = sac_agent.ac.pi.net(state)
            policy_mu = sac_agent.ac.pi.mu_layer(net_out)
            policy_log_std = torch.clamp(sac_agent.ac.pi.log_std_layer(net_out), LOG_STD_MIN, LOG_STD_MAX)
            policy_std = torch.exp(policy_log_std)



            with torch.no_grad():
                prev_net_out = prev_model.net(state)
                prev_mu = prev_model.mu_layer(prev_net_out)
                prev_log_std = torch.clamp(prev_model.log_std_layer(prev_net_out), LOG_STD_MIN, LOG_STD_MAX)
                prev_std = torch.exp(prev_log_std)


                reference_chosen_logps = prev_model.log_prob(state, chosen_act)
                reference_rejected_logps = prev_model.log_prob(reject_state, reject_act)

            kl_divergence_chosen = gaussian_kl_divergence(policy_mu, policy_std, prev_mu, prev_std)


            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            reject_logratios = policy_rejected_logps - reference_rejected_logps

            positive_reward = chosen_logratios.detach().mean().item()
            negative_reward = reject_logratios.detach().mean().item()
            margin = positive_reward - negative_reward

            total_positive_reward += positive_reward
            total_negative_reward += negative_reward
            total_margin += margin

            chosen_losses = 1 - torch.sigmoid(beta * (chosen_logratios - kl_divergence_chosen))
            rejected_losses = 1 - torch.sigmoid(beta * (kl_divergence_rejected - reject_logratios))

            losses = torch.cat((desirable_weight * chosen_losses, undesirable_weight * rejected_losses), 0)
            loss = losses.mean()

            total_loss += loss.item()

            sac_agent.pi_optimizer.zero_grad()
            loss.backward()
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(sac_agent.ac.pi.parameters(), max_norm=1.0)
            sac_agent.pi_optimizer.step()

    return total_loss, total_margin, total_positive_reward, total_negative_reward

def SPPO(sac_agent, prev_model, expert_states, expert_actions, greedy=False, steps=100, eta=1e3, random=False, clip_grad=False):
    assert expert_states.shape[0] == expert_actions.shape[0]
    prev_model.eval()
    batch_size = 256
    total_loss = 0
    total_margin = 0
    total_positive_reward = 0
    total_negative_reward = 0
    desirable_weight = 1.0  # You can adjust this weight
    undesirable_weight = 1.0  # You can adjust this weight
    LOG_STD_MIN = -20
    LOG_STD_MAX = 2
    epsilon = 1e-3
    for i in range(steps):
            # state =
            # chosen_act =
            # reject_act =
            if random:
                reject_act = torch.FloatTensor(np.random.uniform(-1, 1, chosen_act.shape)).to(sac_agent.device)
            else:
                with torch.no_grad():
                    reject_act, reference_rejected_logps = prev_model(reject_state, deterministic=False, with_logprob=True)


            # Clamp the reject action to the action space
            chosen_act = torch.clamp(chosen_act, -1+epsilon, 1-epsilon)
            reject_act = torch.clamp(reject_act, -1+epsilon, 1-epsilon)

            # Calculate log probabilities for chosen actions
            policy_chosen_logps = sac_agent.ac.pi.log_prob(state, chosen_act)
            policy_rejected_logps = sac_agent.ac.pi.log_prob(state, reject_act)

            with torch.no_grad():


                reference_chosen_logps = prev_model.log_prob(state, chosen_act)
                reference_rejected_logps = prev_model.log_prob(state, reject_act)


            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            reject_logratios = policy_rejected_logps - reference_rejected_logps
            
            positive_reward = chosen_logratios.detach().mean().item()
            negative_reward = reject_logratios.detach().mean().item()
            margin = positive_reward - negative_reward
            
            total_positive_reward += positive_reward
            total_negative_reward += negative_reward
            total_margin += margin


            # SPPO loss
            losses = (chosen_logratios-eta/2).pow(2)+ (reject_logratios+eta/2).pow(2)

            loss = losses.mean()
            total_loss += loss.sum()

            sac_agent.pi_optimizer.zero_grad()
            loss.backward()
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(sac_agent.ac.pi.parameters(), max_norm=1.0)
            sac_agent.pi_optimizer.step()

    return total_loss, total_margin, total_positive_reward, total_negative_reward

def SimPO(sac_agent, expert_states, expert_actions, greedy=False,steps=100,beta=2.0,gamma=1,random=False,clip_grad=False):
    assert expert_states.shape[0] == expert_actions.shape[0]
    batch_size = 256
    total_loss = 0
    total_margin = 0
    total_positive_reward = 0
    total_negative_reward = 0
    LOG_STD_MIN = -20
    LOG_STD_MAX = 2
    epsilon = 1e-3

    for i in range(steps):
            # state =
            # chosen_act =
            if random:
                 reject_act = torch.FloatTensor(np.random.uniform(-1,1,chosen_act.shape)).to(sac_agent.device)
            else:
                with torch.no_grad():

                    reject_act, reference_rejected_logps = sac_agent.ac.pi(state, deterministic=False, with_logprob=True)
            # Clamp the reject action to the action space
            chosen_act = torch.clamp(chosen_act, -1+epsilon, 1-epsilon)
            reject_act = torch.clamp(reject_act, -1+epsilon, 1-epsilon)

            # Calculate log probabilities for chosen actions
            chosen_logratios = sac_agent.ac.pi.log_prob(state, chosen_act)
            reject_logratios = sac_agent.ac.pi.log_prob(state, reject_act)


            positive_reward = chosen_logratios.detach().mean().item()
            negative_reward = reject_logratios.detach().mean().item()
            margin = positive_reward - negative_reward
            
            total_positive_reward += positive_reward
            total_negative_reward += negative_reward
            total_margin += margin


            losses = -torch.nn.functional.logsigmoid(beta * chosen_logratios-beta*reject_logratios-gamma)

            loss = losses.mean()
            total_loss += loss.sum()

            sac_agent.pi_optimizer.zero_grad()
            loss.backward()
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(sac_agent.ac.pi.parameters(), max_norm=1.0)
            sac_agent.pi_optimizer.step()

    return total_loss, total_margin, total_positive_reward, total_negative_reward
