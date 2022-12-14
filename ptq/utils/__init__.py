def visualize_per_layer(param, title='test'):
    import matplotlib.pyplot as plt
    channel = 0
    param_list = []
    for idx in range(param.shape[channel]):
        param_list.append(param[idx].cpu().numpy().reshape(-1))

    fig7, ax7 = plt.subplots()
    ax7.set_title(title)
    ax7.boxplot(param_list, showfliers=False)
    plt.show()
