import csv
import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import torch
import matplotlib.animation as animation
from pathlib import Path
from fractions import Fraction

plt.rcParams.update({'font.size': 11, 'figure.figsize': (18 / 2.54, 9 / 2.54)})
line_styles = ['k-o', 'k-^', 'kx-.', 'ko--', 'k:', 'kx-']

td_present = lambda fltr: \
    True if fltr[0][0] is not None and fltr[1][0] is not None \
    else False
fd_present = lambda fltr: \
    True if fltr[0][1] is not None and fltr[1][1] is not None \
    else False

time_c = lambda signal_length, sps: np.arange(signal_length * sps) / sps
time_ac = lambda signal_length, sps: np.arange(-1*signal_length * sps//2, signal_length * sps//2) / sps
omega = lambda signal_length, sps: np.arange(-1*signal_length * sps//2, signal_length * sps//2) \
    / signal_length



def plot_average_num_bits_sweep(params, average_num_bits_all_runs, output_path):
    """
    Plot average number of bits over training iterations
    :param params: Dictionary of parameters
    :param average_num_bits_all_runs:  List of average number of bits of each iteration
    :param output_path: Output directory
    :return:
    """

    plt.clf()

    colors = cm.rainbow(np.linspace(0, 1, len(average_num_bits_all_runs)))

    for i, sweep in enumerate(average_num_bits_all_runs):
        for j, run in enumerate(sweep):
            if j == 0:
                sweep_label = str(params["sweep"]["parameter"][0]) + ": " + str(params["sweep"]["values"][0][i])
                label = "{}, params".format(sweep_label)
                plt.plot(np.arange(len(run["params"])), torch.stack(run["params"]).cpu().detach().numpy(), linestyle="-", color=colors[i], label=label)
                label = "{}, act".format(sweep_label)
                plt.plot(np.arange(len(run["act"])), torch.stack(run["act"]).cpu().detach().numpy(), linestyle="--", color=colors[i], label=label)
            else:
                label = "_nolegend_"
                plt.plot(np.arange(len(run["params"])), torch.stack(run["params"]).cpu().detach().numpy(), linestyle="-", color=colors[i], label=label)
                plt.plot(np.arange(len(run["act"])), torch.stack(run["act"]).cpu().detach().numpy(), linestyle="--", color=colors[i], label=label)

    if params["quantize"] is True:
        plt.axvline(x=params["fp_iterations"], color='g', linestyle='--', label="Start Quant")
        plt.axvline(x=params["fp_iterations"] + params["quant_iterations"], color='y', linestyle='--', label="Start Finetune")

    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")

    plt.xlabel(r'Training itaration')
    plt.ylabel(r'Num Bits')
    plt.grid('on')

    plt.savefig(os.path.join(output_path, "average_quant_bits.png"), bbox_inches='tight')
    plt.close()

    return


def plot_acc_vs_bits_sweep(params, average_test_acc, average_num_bits, output_path):
    """
    Plot accuracy and number of bits of each trained model of sweep
    :param params: Dictionary of parameters
    :param average_test_acc: List of test accuracy of each model
    :param average_num_bits: List of average number of bits of each model
    :param output_path: Output directory
    :return:
    """

    csv_file = open(os.path.join(output_path, "acc_vs_bits.csv"), 'w')
    csv_write = csv.writer(csv_file, delimiter=';')
    csv_write.writerow(['Quant Loss Fac Index', 'Quant Loss Fac', 'Avg Bits Params', 'Avg Bits Acts', 'Acc'])

    plt.clf()
    fig, axs = plt.subplots(nrows=2, sharex=True, sharey=True)
    fig.tight_layout()

    fig.text(0.5, -0.01, 'Average num bits', ha='center')
    fig.text(-0.01, 0.5, 'Average test acc', va='center', rotation='vertical')

    axs[0].set_title("Parameters")
    axs[1].set_title("Activations")

    colors = cm.rainbow(np.linspace(0, 1, len(average_test_acc)))

    for sweep_iter in range(len(average_test_acc)):
        for run_iter in range(params["training_runs"]):

            if run_iter == 0:
                sweep_label = str(params["sweep"]["parameter"][0])+ ": " + str(params["sweep"]["values"][0][sweep_iter])
            else:
                sweep_label = "_nolegend_"

            avg_params_bits = average_num_bits[sweep_iter][run_iter]["params"].detach().cpu().numpy()
            avg_act_bits = average_num_bits[sweep_iter][run_iter]["act"].detach().cpu().numpy()

            current_row = [sweep_iter, params["sweep"]["values"][0][sweep_iter], avg_params_bits.item(), avg_act_bits.item(), average_test_acc[sweep_iter][run_iter]]
            csv_write.writerow(current_row)

            axs[0].scatter(avg_params_bits, average_test_acc[sweep_iter][run_iter], color=colors[sweep_iter], label=sweep_label)
            axs[1].scatter(avg_act_bits, average_test_acc[sweep_iter][run_iter], color=colors[sweep_iter], label=sweep_label)

    csv_file.close()
    # plt.legend(bbox_to_anchor=(1.04,1), loc="upper center")
    plt.figlegend(loc='center left', bbox_to_anchor=(1, 0.5))

    axs[0].set_yscale('log')
    axs[1].set_yscale('log')

    axs[0].grid('on')
    axs[1].grid('on')

    axs[0].ticklabel_format(axis='x', useOffset=False, style='plain')
    axs[1].ticklabel_format(axis='x', useOffset=False, style='plain')

    plt.savefig(os.path.join(output_path, "acc_vs_bits.png"), bbox_inches='tight')
    plt.close()

    return


def write_average_num_bits_per_layer(params, avg_num_bits_per_layer, output_path):
    """
    Write number of bits of each layer to csv file for each trained model
    :param params: Dictionary of parameters
    :param avg_num_bits_per_layer: List of number of bits of each layer of each model
    :param output_path: Output directory
    :return:
    """

    csv_file = open(os.path.join(output_path, "average_num_bits_per_layer.csv"), 'w')
    csv_write = csv.writer(csv_file)

    header_1 = ["Quant Loss Fac"]
    header_2 = [""]
    for layer_index in range(params["depth"]):
        header_1.extend(["", ""])
        header_1.extend(["Layer", str(layer_index)])
        header_1.extend(["", ""])

        header_2.extend(["w int bits", "w frac bits"])
        if "use_bias" not in params or params["use_bias"] is True:
            header_2.extend(["b int bits", "b frac bits"])
        header_2.extend(["a int bits", "a frac bits"])

    csv_write.writerow(header_1)
    csv_write.writerow(header_2)

    for sweep_iter in range(len(avg_num_bits_per_layer)):
        for run_iter in range(params["training_runs"]):
            current_row = [params["sweep"]["values"][0][sweep_iter]]

            for layer_index in range(params["depth"]):
                avg_num_bits_per_layer_curr_model = avg_num_bits_per_layer[sweep_iter][run_iter]

                current_row.append(avg_num_bits_per_layer_curr_model[layer_index]["weights"]["int"])
                current_row.append(avg_num_bits_per_layer_curr_model[layer_index]["weights"]["frac"])

                if "use_bias" not in params or params["use_bias"] is True:
                    current_row.append(avg_num_bits_per_layer_curr_model[layer_index]["bias"]["int"])
                    current_row.append(avg_num_bits_per_layer_curr_model[layer_index]["bias"]["frac"])

                current_row.append(avg_num_bits_per_layer_curr_model[layer_index]["act"]["int"])
                current_row.append(avg_num_bits_per_layer_curr_model[layer_index]["act"]["frac"])

            csv_write.writerow(current_row)

    csv_file.close()

    return


def np_relu(x):
    """
    Relu function
    :param x: input
    :return: output
    """
    return x * (x > 0)



slice_from_center = lambda x, length: x[int(len(x)/2-length/2):int(len(x)/2+length/2)] if length < len(x) else None


def plot_retraining_results(params, channel_changes, ber_no_change, bers_baseline, bers_no_retrain, bers_sup_cont, bers_unsup_cont, bers_sup_step, bers_unsup_step):
    """
    Plot results of retraining
    :param params: Dictionary of parameters
    :param channel_changes:
    :param ber_no_change:
    :param bers_baseline:
    :param bers_no_retrain:
    :param bers_sup_cont:
    :param bers_unsup_cont:
    :param bers_sup_step:
    :param bers_unsup_step:
    :return:
    """

    print("plotting retraining results to {}".format(params["output_files_path"]))

    csv_file = open(os.path.join(params["output_files_path"], "retraining_results.csv"), 'w')
    csv_write = csv.writer(csv_file, delimiter=';')
    csv_write.writerow([params["change_param_name"]] + [str(val) for val in channel_changes])

    csv_write.writerow(["Baseline", str(ber_no_change)] + [str(val) for val in bers_baseline])
    csv_write.writerow(["No retrain", str(ber_no_change)] + [str(val) for val in bers_no_retrain])
    if bers_sup_cont:
        csv_write.writerow(["sup cont", str(ber_no_change)] + [str(val) for val in bers_sup_cont])
    if bers_unsup_cont:
        csv_write.writerow(["unsup cont", str(ber_no_change)] + [str(val) for val in bers_unsup_cont])
    if bers_sup_step:
        csv_write.writerow(["sup step", str(ber_no_change)] + [str(val) for val in bers_sup_step])
    if bers_unsup_step:
        csv_write.writerow(["unsup step", str(ber_no_change)] + [str(val) for val in bers_unsup_step])

    csv_file.close()

    plt.clf()

    x_axis = channel_changes

    plt.plot(x_axis, [ber_no_change] + bers_baseline, label="Baseline", marker="*")
    plt.plot(x_axis, [ber_no_change] + bers_no_retrain, label="No retrain", marker="+")
    if bers_sup_cont:
        plt.plot(x_axis, [ber_no_change] + bers_sup_cont, label="Sup cont", marker="x")
    if bers_unsup_cont:
        plt.plot(x_axis, [ber_no_change] + bers_unsup_cont, label="Unsup cont", marker="x")
    if bers_sup_step:
        plt.plot(x_axis, [ber_no_change] + bers_sup_step, label="Sup step", marker="o")
    if bers_unsup_step:
        plt.plot(x_axis, [ber_no_change] + bers_unsup_step, label="Unsup step", marker="o")

    plt.xlabel(params["change_param_name"])
    if channel_changes[0] > channel_changes[-1]:
        plt.gca().invert_xaxis()

    plt.ylabel("BER")
    plt.legend(loc="upper left")

    plt.yscale('log')
    # plt.grid('on')

    plt.savefig(os.path.join(params["output_files_path"], "retraining_results.png"), bbox_inches='tight')
    plt.close()


def plot_training_accuracies(eval_accs, selected_pumps_accs, params):

    training_iters = params["training_iters"]
    output_dir = params["output_files_path"]

    plt.clf()
    num_training_runs = len(eval_accs)
    data_points = len(eval_accs[0])
    track_loss_interval = int(training_iters // data_points)

    # Set up the X-axis values (training iteration)
    x = [i * track_loss_interval for i in range(data_points - 1)]

    color_map = plt.get_cmap("viridis")

    csv_file = open(os.path.join(output_dir, "training_accuracies.csv"), 'w')
    csv_write = csv.writer(csv_file, delimiter=';')

    for i in range(num_training_runs):
        csv_write.writerow([f'Training Run {i + 1}'])

        color = color_map(i / num_training_runs)

        if len(eval_accs) != 0:
            y = eval_accs[i][:-1].detach().cpu().numpy()
            plt.plot(x, y, color=color, label=f'Training Run {i + 1}')
            csv_write.writerow(["Training iteration"] + [str(val) for val in x])
            csv_write.writerow(["Training eval accs"] + [str(val) for val in y.tolist()])
            if selected_pumps_accs[i] is not None:
                y = selected_pumps_accs[i][:-1].detach().cpu().numpy()
                plt.plot(x, y, color=color, linestyle="--")
                csv_write.writerow(["Training selected pumps accs"] + [str(val) for val in y.tolist()])

    csv_file.close()

    # Add labels and a legend
    plt.xlabel('Training Iteration')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig(os.path.join(output_dir, "training_accuracies.png"), bbox_inches='tight')
    plt.close()


def plot_animated_histograms(losses_0_list, losses_1_list, t, file_path, training_iterations_per_time_step=200, max_x_val=None):

    if isinstance(t, list):
        t_list = t
    else:
        t_list = [t] * len(losses_0_list)

    plt.clf()
    fig, ax = plt.subplots()
    # fig, ax = plt.plot()

    print("len(losses_0_list): ", len(losses_0_list))
    print("len(losses_1_list): ", len(losses_1_list))

    print("len(losses_0_list[0]): ", len(losses_0_list[0]))
    print("len(losses_1_list[0]): ", len(losses_1_list[0]))

    # Flat list of lists of losses_0 and losses_1 to one huge list to calculate mean
    all_losses_list = [item for sublist in losses_0_list for item in sublist] + [item for sublist in losses_1_list for item in sublist]

    #print("losses_0_list: ", losses_0_list)
    #print("losses_1_list: ", losses_1_list)

    def update_hist(num):
        ax.clear()
        # ax[1].clear()

        # Set label of x and y axis
        ax.set_xlabel('MSE Loss')

        # Set max value of y axis
        ax.set_ylim([0, 100])

        # Convert losses_0_list[num] to np array if it is not already
        if isinstance(losses_0_list[num], list):
            losses_0_list[num] = np.array(losses_0_list[num])

        # Convert losses_1_list[num] to np array if it is not already
        if isinstance(losses_1_list[num], list):
            losses_1_list[num] = np.array(losses_1_list[num])

        if max_x_val is not None:
            losses_0 = np.clip(losses_0_list[num], 0, max_x_val)
            losses_1 = np.clip(losses_1_list[num], 0, max_x_val)
        else:
            losses_0 = losses_0_list[num]
            losses_1 = losses_1_list[num]

        plt.axvline(x=t_list[num], color='black', label='threshold')

        if max_x_val is not None:
            if len(losses_0) < len(losses_1):
                ax.hist(losses_1, bins=200, range=[0, max_x_val], color='red', label='Label 1')
                ax.hist(losses_0, bins=200, range=[0, max_x_val], color='blue', label='Label 0')
            else:
                ax.hist(losses_0, bins=200, range=[0, max_x_val], color='blue', label='Label 0')
                ax.hist(losses_1, bins=200, range=[0, max_x_val], color='red', label='Label 1')
        else:
            if len(losses_0) < len(losses_1):
                ax.hist(losses_1, bins=200, color='red', label='Label 1')
                ax.hist(losses_0, bins=200, color='blue', label='Label 0')
            else:
                ax.hist(losses_0, bins=200, color='blue', label='Label 0')
                ax.hist(losses_1, bins=200, color='red', label='Label 1')

        # ax[0].set_title('losses_0')
        # ax[0].set_title('losses_1')

        # Count true positive rate
        tp = np.sum(losses_1 > t_list[num])
        fp = np.sum(losses_0 > t_list[num])
        tn = np.sum(losses_0 < t_list[num])
        fn = np.sum(losses_1 < t_list[num])

        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        tnr = tn / (tn + fp)
        fnr = fn / (fn + tp)

        # Show legend
        ax.legend()
        fig.suptitle('Iteration: {} | tpr: {:.2f} | fpr: {:.2f} | tnr: {:.2f} | fnr: {:.2f}'.format(num * training_iterations_per_time_step, tpr, fpr, tnr, fnr))

    ani = animation.FuncAnimation(fig, update_hist, len(losses_0_list))

    file_extension = Path(file_path).suffix

    # ffmpeg needs to be available to save as mp4
    if file_extension == '.mp4' and animation.writers.is_available('ffmpeg'):
        ani.save(file_path)
    else:
        gif_filename = Path(file_path).with_suffix('.gif')
        ani.save(gif_filename)

    # Save last frame as png
    update_hist(len(losses_0_list) - 2)
    png_filename = Path(file_path).with_suffix('.png')
    plt.savefig(png_filename)

    plt.close(fig)


def plot_animated_vibration_data(in_0, in_1, out_0, out_1, file_path):
    """
    Plot numpy array of vibration data of size 3 x 800. One plot for each dimension x,y and z.
    Plot in_0 and out_0 in one plot and in_1 and out_1 in another plot.
    :param in_0: list of input samples corresponding to class 0
    :param in_1: list of input samples corresponding to class 1
    :param out_0: list of output samples corresponding to class 0
    :param out_1: list of output samples corresponding to class 1
    :return:
    """

    plt.clf()

    fig, axs = plt.subplots(3, 2, constrained_layout=True)

    def update_plot(num):
        axs[0, 0].clear()
        axs[0, 0].plot(in_0[num][0], label="in_0")
        axs[0, 0].plot(out_0[num][0], label="out_0")
        axs[0, 0].set_title("x")
        axs[0, 0].legend()

        axs[1, 0].clear()
        axs[1, 0].plot(in_0[num][1], label="in_0")
        axs[1, 0].plot(out_0[num][1], label="out_0")
        axs[1, 0].set_title("y")

        axs[2, 0].clear()
        axs[2, 0].plot(in_0[num][2], label="in_0")
        axs[2, 0].plot(out_0[num][2], label="out_0")
        axs[2, 0].set_title("z")

        axs[0, 1].clear()
        axs[0, 1].plot(in_1[num][0], label="in_1")
        axs[0, 1].plot(out_1[num][0], label="out_1")
        axs[0, 1].set_title("x")
        axs[0, 1].legend()

        axs[1, 1].clear()
        axs[1, 1].plot(in_1[num][1], label="in_1")
        axs[1, 1].plot(out_1[num][1], label="out_1")
        axs[1, 1].set_title("y")

        axs[2, 1].clear()
        axs[2, 1].plot(in_1[num][2], label="in_1")
        axs[2, 1].plot(out_1[num][2], label="out_1")
        axs[2, 1].set_title("z")

    ani = animation.FuncAnimation(fig, update_plot, len(in_0))

    file_extension = Path(file_path).suffix

    # ffmpeg needs to be available to save as mp4
    if file_extension == '.mp4' and animation.writers.is_available('ffmpeg'):
        ani.save(file_path, dpi=400)
    else:
        gif_filename = Path(file_path).with_suffix('.gif')
        ani.save(gif_filename, dpi=400)

    # Save last frame as png
    update_plot(len(in_0) - 1)
    png_filename = Path(file_path).with_suffix('.png')
    plt.savefig(png_filename, dpi=400)


def plot_param_sweep(param_list, acc_list, tp_list, fp_list, tn_list, fn_list, output_path):

    # Calc true_positive_rate
    tpr_list = [tp_list[i] / (tp_list[i] + fn_list[i]) if tp_list[i] + fn_list[i] != 0 else 0 for i in range(len(tp_list))]
    # Calc false_positive_rate
    fpr_list = [fp_list[i] / (fp_list[i] + tn_list[i]) if fp_list[i] + tn_list[i] != 0 else 0 for i in range(len(fp_list))]
    # Calc true_negative_rate
    tnr_list = [tn_list[i] / (tn_list[i] + fp_list[i]) if tn_list[i] + fp_list[i] != 0 else 0 for i in range(len(tn_list))]
    # Calc false_negative_rate
    fnr_list = [fn_list[i] / (fn_list[i] + tp_list[i]) if fn_list[i] + tp_list[i] != 0 else 0 for i in range(len(fn_list))]

    plt.clf()

    plt.plot(acc_list, label='Accuracy')
    plt.plot(tpr_list, label='True Positive Rate')
    plt.plot(fpr_list, label='False Positive Rate')
    plt.plot(tnr_list, label='True Negative Rate')
    plt.plot(fnr_list, label='False Negative Rate')

    # Set labels of x axis to add_in_fac_list converted to string
    plt.xticks(np.arange(len(param_list)), ["{:.1E}".format(add_in_fac) for add_in_fac in param_list], fontsize=6, rotation=60)

    if 1 in param_list:
        # Get index of add_in_fac_list where add_in_fac is 1
        index = param_list.index(1)

        # Plot vertical line at x=1
        plt.axvline(x=index, color='black', linestyle='--')

    plt.xlabel('Param')
    plt.legend()
    plt.autoscale()

    plt.savefig(output_path, dpi=400)

    # Also save as csv
    # Remove file extension from output_path and replace with .csv
    csv_path = Path(output_path).with_suffix('.csv')
    csv_file = open(csv_path, 'w')
    csv_write = csv.writer(csv_file, delimiter=';')

    csv_write.writerow(['Add In Fac', 'Accuracy', 'True Positive Rate', 'False Positive Rate', 'True Negative Rate', 'False Negative Rate'])

    for i in range(len(param_list)):
        csv_write.writerow([param_list[i], acc_list[i], tpr_list[i], fpr_list[i], tnr_list[i], fnr_list[i]])

    csv_file.close()












