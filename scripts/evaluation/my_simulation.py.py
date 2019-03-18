def get_fake_trajectories(obs_traj, obs_traj_rel):
    x1 = torch.ones(8).cuda() * 5.5 #torch.range(.5, 4, .5).cuda()
    y1 = torch.range(.5, 4, .5).cuda()
    x2 = torch.ones(8).cuda() * 3.5#torch.range(9, 5.5, -.5).cuda()
    y2 = torch.range(.5, 4, .5).cuda()
    x3 = torch.ones(8).cuda() * 4.5
    y3 = torch.range(9.5, 6, -.5).cuda()

    X1 = torch.stack([x1, y1])
    X2 = torch.stack([x2, y2])
    X3 = torch.stack([x3, y3])

    XX = torch.stack([X1, X2, X3]).permute(2, 0, 1)
    Xdelta = torch.zeros_like(XX)

    for i in range(XX.size(1)):
        obs_traj[:, i] = XX[:, i]
        Xdelta[1:, i] = XX[1:, i, :] - XX[:-1, i, :]
        obs_traj_rel[:, i] = Xdelta[:, i]
    return obs_traj, obs_traj_rel

def plot_prediction(obs_traj, pred_traj_fake1, pred_traj_fake2, ax3, ax4):
    for i in range(3):
        ax3.plot(obs_traj[:, i, 0].cpu().numpy(), obs_traj[:, i, 1].cpu().numpy(), color=colors[i], marker='o', markersize=2)
        ax3.plot(pred_traj_fake1[:, i, 0].cpu().numpy(), pred_traj_fake1[:, i, 1].cpu().numpy(), color=colors[i], marker='o', markersize=2, linestyle='--')
        ax3.quiver(pred_traj_fake1[0, i, 0].cpu().numpy(), pred_traj_fake1[0, i, 1].cpu().numpy(), pred_traj_fake1[1, i, 0].cpu().numpy() - pred_traj_fake1[0, i, 0].cpu().numpy(), pred_traj_fake1[1, i, 1].cpu().numpy() - pred_traj_fake1[0, i, 1].cpu().numpy())
        ax3.axis([0, 10, 0, 10])
        ax4.plot(obs_traj[:, i, 0].cpu().numpy(), obs_traj[:, i, 1].cpu().numpy(), color=colors[i], marker='o', markersize=1)
        ax4.quiver(pred_traj_fake2[0, i, 0].cpu().numpy(), pred_traj_fake2[0, i, 1].cpu().numpy(),
                   pred_traj_fake2[1, i, 0].cpu().numpy() - pred_traj_fake2[0, i, 0].cpu().numpy(),
                   pred_traj_fake2[1, i, 1].cpu().numpy() - pred_traj_fake2[0, i, 1].cpu().numpy())
        ax4.plot(pred_traj_fake2[:, i, 0].cpu().numpy(), pred_traj_fake2[:, i, 1].cpu().numpy(), color=colors[i], marker='o', markersize=1, linestyle='--')
        ax4.axis([0, 10, 0, 10])
    plt.show()


