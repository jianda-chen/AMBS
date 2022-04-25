# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia

import utils
from sac_ae_ratio import  Actor, Critic, LOG_FREQ
from transition_model import make_transition_model


class AMBSRatioAgent(object):
    """Bisimulation metric algorithm."""
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        transition_model_type,
        hidden_dim=256,
        discount=0.99,
        init_temperature=0.01,
        alpha_lr=1e-3,
        alpha_beta=0.9,
        actor_lr=1e-3,
        actor_beta=0.9,
        actor_log_std_min=-10,
        actor_log_std_max=2,
        actor_update_freq=2,
        encoder_stride=2,
        critic_lr=1e-3,
        critic_beta=0.9,
        critic_tau=0.005,
        critic_target_update_freq=2,
        encoder_type='pixel',
        encoder_feature_dim=50,
        encoder_lr=1e-3,
        encoder_tau=0.005,
        decoder_type='pixel',
        decoder_lr=1e-3,
        decoder_update_freq=1,
        decoder_latent_lambda=0.0,
        decoder_weight_lambda=0.0,
        num_layers=4,
        num_filters=32,
        bisim_coef=0.5,
        sep_rew_dyn=False,
        sep_rew_ratio=0.5,
        adaptive_ratio=False,
        deep_metric=False,
    ):
        print(__file__)
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.decoder_update_freq = decoder_update_freq
        self.decoder_latent_lambda = decoder_latent_lambda
        self.transition_model_type = transition_model_type
        self.bisim_coef = bisim_coef
        self.encoder_feature_dim = encoder_feature_dim

        #jd
        self.sep_rew_dyn = sep_rew_dyn
        self.sep_rew_ratio = sep_rew_ratio
        self.deep_metric = deep_metric
        self.adaptive_ratio = adaptive_ratio
        print(self.sep_rew_dyn, self.sep_rew_ratio, self.deep_metric, self.adaptive_ratio)

        if self.sep_rew_dyn:
            self.rew_length = round(float(encoder_feature_dim) * self.sep_rew_ratio)
            self.dyn_start_idx = self.rew_length
            self.transition_model = make_transition_model(
                transition_model_type, encoder_feature_dim - self.rew_length, action_shape
            ).to(device)
            self.reward_decoder = nn.Sequential(
            nn.Linear(encoder_feature_dim - self.rew_length, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 1)).to(device)
        else:
            self.transition_model = make_transition_model(
                transition_model_type, encoder_feature_dim, action_shape
            ).to(device)
            self.reward_decoder = nn.Sequential(
                nn.Linear(encoder_feature_dim, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Linear(512, 1)).to(device)

        if self.deep_metric:
            self.deep_metric_rew = nn.Sequential(
                nn.Linear(self.rew_length * 2, 50),
                nn.ReLU(),
                nn.Linear(50, 1),
            ).to(device)
            self.deep_metric_dyn = nn.Sequential(
                nn.Linear((encoder_feature_dim - self.rew_length) * 2, 50),
                nn.ReLU(),
                nn.Linear(50, 1),
            ).to(device)

        self.actor = Actor(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, self.rew_length, actor_log_std_min, actor_log_std_max,
            num_layers, num_filters, encoder_stride
        ).to(device)

        self.critic = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, self.rew_length, num_layers, num_filters, encoder_stride
        ).to(device)

        self.critic_target = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, self.rew_length, num_layers, num_filters, encoder_stride
        ).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())


        # tie encoders between actor and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

        # optimizer for decoder
        if self.deep_metric:
            self.decoder_optimizer = torch.optim.Adam(
                list(self.reward_decoder.parameters()) + list(self.transition_model.parameters())
                + list(self.deep_metric_rew.parameters()) + list(self.deep_metric_dyn.parameters()),
                lr=decoder_lr,
                weight_decay=decoder_weight_lambda
            )
        else:
            self.decoder_optimizer = torch.optim.Adam(
                list(self.reward_decoder.parameters()) + list(self.transition_model.parameters()),
                lr=decoder_lr,
                weight_decay=decoder_weight_lambda
            )

        # optimizer for critic encoder for reconstruction loss
        self.encoder_optimizer = torch.optim.Adam(
            self.critic.encoder.parameters(), lr=encoder_lr
        )

        self.actor.encoder.ratio.requires_grad = False
        if not adaptive_ratio:
            self.critic.encoder.ratio.requires_grad = False

        image_pad = 4
        self.aug_trans = self.aug_trans = nn.Sequential(
            nn.ReplicationPad2d(image_pad),
            kornia.augmentation.RandomCrop((obs_shape[-2], obs_shape[-1])))

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    def ratio_softmax(self):
        softmax_ratio = torch.softmax(self.ratio, dim=-1)
        expand_softmax_ratio = torch.repeat_interleave(
            softmax_ratio, torch.tensor(
                (self.rew_length, self.encoder_feature_dim - self.rew_length)))
        return expand_softmax_ratio

    def select_action(self, obs):
        with torch.no_grad():
            # obs = torch.FloatTensor(obs).to(self.device)
            obs = torch.tensor(obs, device=self.device, dtype=torch.float)
            obs = obs.unsqueeze(0)
            mu, _, _, _ = self.actor(
                obs, compute_pi=False, compute_log_pi=False
            )
            return mu.cpu().data.numpy().flatten()

    def select_action_batch(self, obs):
        with torch.no_grad():
            # obs = torch.FloatTensor(obs).to(self.device)
            obs = torch.tensor(obs, device=self.device, dtype=torch.float)
            mu, _, _, _ = self.actor(
                obs, compute_pi=False, compute_log_pi=False
            )
            return mu.cpu().flatten(start_dim=1).data.numpy()

    def sample_action(self, obs):
        with torch.no_grad():
            # obs = torch.FloatTensor(obs).to(self.device)
            obs = torch.tensor(obs, device=self.device, dtype=torch.float)
            obs = obs.unsqueeze(0)
            mu, pi, _, _ = self.actor(obs, compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()

    def sample_action_batch(self, obs):
        with torch.no_grad():
            # obs = torch.FloatTensor(obs).to(self.device)
            obs = torch.tensor(obs, device=self.device, dtype=torch.float)
            mu, pi, _, _ = self.actor(obs, compute_log_pi=False)
            return pi.cpu().data.numpy()

    def update_critic(self, obs, action, reward, next_obs, not_done, aug_obs, aug_next_obs, L, step):
        with torch.no_grad():
            batch_size = next_obs.shape[0]
            next_obs_and_aug = torch.cat((next_obs, aug_next_obs))
            not_done = torch.cat((not_done, not_done), dim=0)
            reward = torch.cat((reward, reward), dim=0)

            _, policy_action, log_pi, _ = self.actor(next_obs_and_aug)
            target_Q1, target_Q2 = self.critic_target(next_obs_and_aug, policy_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

            target_Q = (target_Q[:batch_size] + target_Q[batch_size:]) / 2


        # get current Q estimates
        obs_and_aug = torch.cat((obs, aug_obs), dim=0)
        action = torch.cat((action, action), dim=0)
        target_Q = torch.cat((target_Q, target_Q), dim=0)
        current_Q1, current_Q2 = self.critic(obs_and_aug, action, detach_encoder=False)
        critic_loss = F.mse_loss(current_Q1,
                                 target_Q) + F.mse_loss(current_Q2, target_Q)

        critic_loss = critic_loss / 2.

        L.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), 40.)
        self.critic_optimizer.step()

        self.critic.log(L, step)

    def update_actor_and_alpha(self, obs, L, step):
        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std = self.actor(obs, detach_encoder=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        L.log('train_actor/loss', actor_loss, step)
        L.log('train_actor/target_entropy', self.target_entropy, step)
        entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)
                                            ) + log_std.sum(dim=-1)
        L.log('train_actor/entropy', entropy.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), 40.)
        self.actor_optimizer.step()

        self.actor.log(L, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_pi - self.target_entropy).detach()).mean()
        L.log('train_alpha/loss', alpha_loss, step)
        L.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update_encoder(self, obs, action, reward, aug_obs, aug_next_obs, L, step):
        _, h = self.critic.encoder(obs, pre_ratio=True)

        # Sample random states across episodes at random
        batch_size = obs.size(0)
        perm = torch.from_numpy(np.random.permutation(batch_size))
        h2 = h[perm]

        # augmentation
        _, aug_h = self.critic.encoder(aug_obs, pre_ratio=True)

        if self.sep_rew_dyn:
            h_rew, h_dyn = h[..., :self.rew_length], h[..., self.rew_length:]
            h2_rew, h2_dyn = h2[..., :self.rew_length], h2[..., self.rew_length:]   
            aug_h_rew, aug_h_dyn = aug_h[..., :self.rew_length], aug_h[..., self.rew_length:]


        with torch.no_grad():
            action, _, _, _ = self.actor(obs, compute_pi=False, compute_log_pi=False)
            if self.sep_rew_dyn:
                pred_next_latent_mu1, pred_next_latent_sigma1 = self.transition_model(torch.cat([h_dyn, action], dim=1))
            else:
                pred_next_latent_mu1, pred_next_latent_sigma1 = self.transition_model(torch.cat([h, action], dim=1))
            # pred_next_latent_mu1, pred_next_latent_sigma1 = self.transition_model(torch.cat([h, action], dim=1))
            # reward = self.reward_decoder(pred_next_latent_mu1)
            reward2 = reward[perm]

        if pred_next_latent_sigma1 is None:
            pred_next_latent_sigma1 = torch.zeros_like(pred_next_latent_mu1)
        
        # augmentation
        if self.sep_rew_dyn:
            h_rew = torch.cat((h_rew, h2_rew), dim=0)
            h_dyn = torch.cat((h_dyn, h2_dyn), dim=0)
            h2_rew = torch.cat((h2_rew, aug_h_rew), dim=0)
            h2_dyn = torch.cat((h2_dyn, aug_h_dyn), dim=0)
            reward = torch.cat((reward, reward), dim=0)
            reward2 = torch.cat((reward2, reward2), dim=0)
            perm = torch.cat((perm, perm), dim=0)
            if pred_next_latent_mu1.ndim == 2:
                pred_next_latent_mu1 = torch.cat((pred_next_latent_mu1, pred_next_latent_mu1), dim=0)
                pred_next_latent_sigma1 = torch.cat((pred_next_latent_sigma1, pred_next_latent_sigma1), dim=0)
            elif pred_next_latent_mu1.ndim == 3: # (E, B, Z)
                pred_next_latent_mu1 = torch.cat((pred_next_latent_mu1, pred_next_latent_mu1), dim=1)
                pred_next_latent_sigma1 = torch.cat((pred_next_latent_sigma1, pred_next_latent_sigma1), dim=1)
            else:
                raise NotImplementedError
        
        if pred_next_latent_mu1.ndim == 2:  # shape (B, Z), no ensemble
            pred_next_latent_mu2 = pred_next_latent_mu1[perm]
            pred_next_latent_sigma2 = pred_next_latent_sigma1[perm]
        elif pred_next_latent_mu1.ndim == 3:  # shape (B, E, Z), using an ensemble
            pred_next_latent_mu2 = pred_next_latent_mu1[:, perm]
            pred_next_latent_sigma2 = pred_next_latent_sigma1[:, perm]
        else:
            raise NotImplementedError       

        if self.sep_rew_dyn:
            softmax_ratio = torch.softmax(self.critic.encoder.ratio.detach(), dim=-1)
            # softmax_ratio = self.critic.encoder.softmax_ratio.detach()
            if self.deep_metric:
                rew_dist = softmax_ratio[0] * self.deep_metric_rew(torch.cat((h_rew, h2_rew), dim=-1)).pow(2.) # Mar 6
                dyn_dist = softmax_ratio[1] * self.deep_metric_dyn(torch.cat((h_dyn, h2_dyn), dim=-1)).pow(2.) # Mar 6
            else:
                rew_dist = softmax_ratio[0] * F.smooth_l1_loss(h_rew, h2_rew, reduction='none').mean(-1, keepdim=True)
                dyn_dist = F.mse_loss(softmax_ratio[1] * h_dyn, softmax_ratio[1] * h2_dyn, reduction='none').mean(-1, keepdim=True)
            
            r_dist = softmax_ratio[0] * F.smooth_l1_loss(reward, reward2, reduction='none')
            if self.transition_model_type == '':
                transition_dist = softmax_ratio[1] * F.smooth_l1_loss(pred_next_latent_mu1, pred_next_latent_mu2, reduction='none')
            else:
                transition_dist = softmax_ratio[1] * torch.sqrt(
                    (pred_next_latent_mu1 - pred_next_latent_mu2).pow(2) +
                    (pred_next_latent_sigma1 - pred_next_latent_sigma2).pow(2)
                ).mean(-1, keepdim=True)

            loss = .5 * (rew_dist - r_dist).pow(2).mean() + \
                    .5 * (dyn_dist - self.discount * transition_dist).pow(2).mean()
            with torch.no_grad():
                L.log('train_ae/encoder_loss_dyn', 
                    .5 * (rew_dist - r_dist).pow(2).mean().item(), 
                    step)
                L.log('train_ae/encoder_loss_rew', 
                    .5 * (rew_dist - r_dist).pow(2).mean().item(),
                    step)
                L.log('train_ae/encoder_dyn_dist', 
                    torch.sqrt(
                        (pred_next_latent_mu1 - pred_next_latent_mu2).pow(2) +
                        (pred_next_latent_sigma1 - pred_next_latent_sigma2).pow(2)
                    ).mean().item(), step)
                L.log('train_ae/encoder_dyn_mu', 
                    pred_next_latent_mu1.abs().mean().item(), step)
                L.log('train_ae/encoder_dyn_sigma', 
                    pred_next_latent_sigma1.abs().mean().item(), step)
                L.log('train_ae/encoder_h_dyn', 
                    h_dyn.abs().mean().item(), step)
                L.log('train_ae/encoder_h_rew', 
                    h_rew.abs().mean().item(), step)

        else:
            raise NotImplementedError

        L.log('train_ae/encoder_loss', loss, step)
        return loss

    def update_transition_reward_model(self, obs, action, next_obs, reward, L, step, aug=False):
        _, h = self.critic.encoder(obs, pre_ratio=True)
        if self.sep_rew_dyn:
            h_rew, h_dyn = h[..., :self.rew_length], h[..., self.rew_length:]
        if self.sep_rew_dyn:
            pred_next_latent_mu, pred_next_latent_sigma = self.transition_model(torch.cat([h_dyn, action], dim=1))
        else:
            pred_next_latent_mu, pred_next_latent_sigma = self.transition_model(torch.cat([h, action], dim=1))
        # pred_next_latent_mu, pred_next_latent_sigma = self.transition_model(torch.cat([h, action], dim=1))
        if pred_next_latent_sigma is None:
            pred_next_latent_sigma = torch.ones_like(pred_next_latent_mu)

        _, next_h = self.critic.encoder(next_obs, pre_ratio=True)
        if self.sep_rew_dyn:
            next_h_rew, next_h_dyn = next_h[..., :self.rew_length], next_h[..., self.rew_length:]
            diff = (0.5 * pred_next_latent_mu - 0.5 * next_h_dyn.detach()) / pred_next_latent_sigma
        else:
            diff = (pred_next_latent_mu - next_h.detach()) / pred_next_latent_sigma
        loss = torch.mean(0.5 * diff.pow(2) + torch.log(pred_next_latent_sigma)) # Mar 13
        if not aug:
            L.log('train_ae/transition_loss', loss, step)

        if self.sep_rew_dyn:
            pred_next_latent = self.transition_model.sample_prediction(torch.cat([h_dyn, action], dim=1))
            pred_next_reward = self.reward_decoder(pred_next_latent)
            reward_loss = F.mse_loss(pred_next_reward, reward)
            total_loss = loss + reward_loss
        else:
            raise NotImplementedError
        return total_loss

    def update(self, replay_buffer, L, step):
        obs, action, _, reward, next_obs, not_done = replay_buffer.sample()
        

        aug_obs = self.aug_trans(obs)
        aug_next_obs = self.aug_trans(next_obs)

        obs = self.aug_trans(obs)
        next_obs = self.aug_trans(next_obs)
        
        L.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done, aug_obs, aug_next_obs, L, step)
        transition_reward_loss = self.update_transition_reward_model(obs, action, next_obs, reward, L, step)
        encoder_loss = self.update_encoder(obs, action, reward, aug_obs, aug_next_obs, L, step)
        total_loss = self.bisim_coef * encoder_loss + 1e-2 * transition_reward_loss
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        total_loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.encoder, self.critic_target.encoder,
                self.encoder_tau
            )

            
    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic_target.state_dict(), '%s/critic_target_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.reward_decoder.state_dict(),
            '%s/reward_decoder_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.log_alpha,
            '%s/log_alpha_%s.pt' % (model_dir, step)
        )
        if self.deep_metric:
            torch.save(
                self.deep_metric_dyn.state_dict(),
                '%s/deep_metric_dyn_%s.pt' % (model_dir, step)
            )
            torch.save(
                self.deep_metric_rew.state_dict(),
                '%s/deep_metric_rew_%s.pt' % (model_dir, step)
            )
        torch.save(
            self.transition_model.state_dict(),
            '%s/transition_model_%s.pt' % (model_dir, step)
        )

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )

