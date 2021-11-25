import torch

from tqc.functions import quantile_huber_loss_f
from tqc import DEVICE


class Trainer(object):
    def __init__(
            self,
            *,
            actor,
            critic,
            critic_target,
            discount,
            tau,
            top_quantiles_to_drop,
            target_entropy,
            use_acc,
            lr_dropped_quantiles,
            adjusted_dropped_quantiles_init,
            adjusted_dropped_quantiles_max,
            diff_ma_coef,
            num_critic_updates,
            writer
    ):

        self.actor = actor
        self.critic = critic
        self.critic_target = critic_target
        self.log_alpha = torch.zeros((1,), requires_grad=True, device=DEVICE)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)

        self.discount = discount
        self.tau = tau
        self.top_quantiles_to_drop = top_quantiles_to_drop
        self.target_entropy = target_entropy

        self.quantiles_total = critic.n_quantiles * critic.n_nets

        self.total_it = 0

        self.writer = writer

        self.use_acc = use_acc
        self.num_critic_updates = num_critic_updates
        if use_acc:
            self.adjusted_dropped_quantiles = torch.tensor(adjusted_dropped_quantiles_init, requires_grad=True)
            self.adjusted_dropped_quantiles_max = adjusted_dropped_quantiles_max
            self.dropped_quantiles_dropped_optimizer = torch.optim.SGD([self.adjusted_dropped_quantiles], lr=lr_dropped_quantiles)
            self.first_training = True
            self.diff_ma_coef = diff_ma_coef

    def train(self, replay_buffer, batch_size=256, ptr_list=None, disc_return=None, do_beta_update=False):

        if ptr_list is not None and do_beta_update:
            self.update_beta(replay_buffer, ptr_list, disc_return)

        for it in range(self.num_critic_updates):
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
            alpha = torch.exp(self.log_alpha)

            # --- Q loss ---
            with torch.no_grad():
                # get policy action
                new_next_action, next_log_pi = self.actor(next_state)
                # compute and cut quantiles at the next state
                next_z = self.critic_target(next_state, new_next_action)  # batch x nets x quantiles
                sorted_z, _ = torch.sort(next_z.reshape(batch_size, -1))
                if self.use_acc:
                    sorted_z_part = sorted_z[:, :self.quantiles_total - round(self.critic.n_nets * self.adjusted_dropped_quantiles.item())]
                else:
                    sorted_z_part = sorted_z[:, :self.quantiles_total - self.top_quantiles_to_drop]

                # compute target
                target = reward + not_done * self.discount * (sorted_z_part - alpha * next_log_pi)

            cur_z = self.critic(state, action)
            critic_loss = quantile_huber_loss_f(cur_z, target.detach())

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


        # --- Policy and alpha loss ---
        new_action, log_pi = self.actor(state)
        alpha_loss = -self.log_alpha * (log_pi + self.target_entropy).detach().mean()
        actor_loss = (alpha * log_pi - self.critic(state, new_action).mean(2).mean(1, keepdim=True)).mean()


        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.total_it += 1

        if self.total_it % 1000 == 0:
            self.writer.add_scalar('learner/critic_loss', critic_loss.detach().cpu().numpy(), self.total_it)
            self.writer.add_scalar('learner/actor_loss', actor_loss.detach().cpu().numpy(), self.total_it)
            self.writer.add_scalar('learner/alpha_loss', alpha_loss.detach().cpu().numpy(), self.total_it)
            self.writer.add_scalar('learner/alpha', alpha.detach().cpu().numpy(), self.total_it)
            self.writer.add_scalar('learner/Q_estimate', cur_z.mean().detach().cpu().numpy(), self.total_it)

    def update_beta(self, replay_buffer, ptr_list=None, disc_return=None):
        state, action = replay_buffer.states_by_ptr(ptr_list)
        disc_return = torch.FloatTensor(disc_return).to(DEVICE)
        assert disc_return.shape[0] == state.shape[0]

        mean_Q_last_eps =  self.critic(state, action).mean(2).mean(1, keepdim=True).mean().detach()
        mean_return_last_eps = torch.mean(disc_return).detach()

        if self.first_training:
            self.diff_mvavg = torch.abs(mean_return_last_eps - mean_Q_last_eps).detach()
            self.first_training = False
        else:
            self.diff_mvavg = (1 - self.diff_ma_coef) * self.diff_mvavg \
                              + self.diff_ma_coef * torch.abs(mean_return_last_eps - mean_Q_last_eps).detach()

        diff_qret = ((mean_return_last_eps - mean_Q_last_eps) / (self.diff_mvavg + 1e-8)).detach()
        aux_loss = self.adjusted_dropped_quantiles * diff_qret
        self.dropped_quantiles_dropped_optimizer.zero_grad()
        aux_loss.backward()
        self.dropped_quantiles_dropped_optimizer.step()
        self.adjusted_dropped_quantiles.data = self.adjusted_dropped_quantiles.clamp(min=0., max=self.adjusted_dropped_quantiles_max)

        self.writer.add_scalar('learner/adjusted_dropped_quantiles', self.adjusted_dropped_quantiles, self.total_it)

    def save(self, filename):
        filename = str(filename)
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_target.state_dict(), filename + "_critic_target")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
        torch.save(self.log_alpha, filename + '_log_alpha')
        torch.save(self.alpha_optimizer.state_dict(), filename + "_alpha_optimizer")

    def load(self, filename):
        filename = str(filename)
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_target.load_state_dict(torch.load(filename + "_critic_target"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.log_alpha = torch.load(filename + '_log_alpha')
        self.alpha_optimizer.load_state_dict(torch.load(filename + "_alpha_optimizer"))
