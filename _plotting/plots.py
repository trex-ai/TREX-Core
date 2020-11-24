
import dataset
import numpy as np
# import plotly.graph_objects as go
import matplotlib.pyplot as plt

def plot_metric(agent_id, db_path, sim_type, metric, gens=None, compare_with_optimal=False):
    """

    :param agent_id: string agent number : eGauge13380
    :param db_path:
    :param sim_type: type of sim : csp or verification
    :param metric: metric to look at
    :param gens: number of gens to plot
    :return:
    """
    if not gens:
        gens = gens_count(db_path, sim_type) # extraire tout les donnees automatiquement
        # print(gens)
    metric_ts = []
    for gen in range(gens):
        print()
        metrics = from_metrics(db_path, gen, sim_type, agent_id)
        # print('what',metrics)
        if gen == 0:
            gen_len = len(metrics[metric])
        if metric not in metrics:
            print('uhoh!')
            break

        metric_ts.extend(metrics[metric])

    # if metric == 'rewards' and compare_with_optimal:
    #     # from _utils.simple_optimal_value import optimal_reward
    #     from analysis.simple_optimal_value2 import optimal_reward
    #     opt_reward = optimal_reward(gen_len-1, agent_id)
    #     # opt_reward = np.tile(opt_reward, gens)
    #     print('opt_reward:' ,len(opt_reward))
    #     print('metrics:', len(metric_ts))

    x_axis_max = len(metric_ts)/gen_len
    x_axis = np.linspace(0, x_axis_max, len(metric_ts))
    # if compare_with_optimal:
    #     for idx in range(1, len(metric_ts)):
    #         o_r = round(opt_reward[idx], 2)
    #         a_r = round(metric_ts[idx], 2)
    #         if o_r != a_r:
    #             print('uh oh', idx, o_r, a_r, o_r - a_r)

    # print(sum(opt_reward[1:]) - sum(metric_ts[1:]))

    # if metric == 'rewards' and compare_with_optimal:
    #     plt.plot(x_axis, (opt_reward*gen)[:len(metric_ts)], label='Theoretical Optimal')
    #     plt.plot(x_axis, metric_ts, label='Actual_optimal')
    # else:
    plt.scatter(x=x_axis, y=metric_ts, s=.5, alpha=0.3, label=metric)
    plt.title(metric + ' for ' + str(gen+1) + ' gens')
    plt.xlabel('Generations')
    plt.ylabel(metric)
    plt.legend()
    plt.show()

def gens_count(db_path, sim_type):
    from sqlalchemy import create_engine
    engine = create_engine(db_path)
    tables = set([int(tbl_name.split('_')[0]) for tbl_name in engine.table_names() if tbl_name[0].isdigit()])
    return max(tables) + 1


def from_metrics(db_path, gen, table_name, agent_id, **kwargs):
    table_name = '_'.join([str(gen), sim_type, 'metrics', agent_id])

    db = dataset.connect(db_path)
    table = db[table_name]

    data = []
    for row in table:
        data.append(row)

    if data:
        return {k: [dic[k] for dic in data] for k in data[0]}
    return {}

# def plot_actions(agent_id, db_path, sim_type, action, param, gens=None, compare_with_optimal=False):
#     # examples of actions can be 'bids', 'asks'
#     # examples of param can be 'price', 'quantity', etc
#
#     """
#
#     :param agent_id: string agent number : eGauge13380
#     :param db_path:
#     :param sim_type: type of sim : csp or verification
#     :param metric: metric to look at
#     :param gens: number of gens to plot
#     :return:
#     """
#     if not gens:
#         gens = extract.gens_count(db_path, sim_type)  # extraire tout les donnees automatiquement
#     metric_ts = []
#     for gen in range(gens):
#         metrics = extract.from_metrics(db_path, gen, sim_type, agent_id)
#         if 'actions_dict' not in metrics:
#             break
#         if gen == 0:
#             gen_len = len(metrics['actions_dict'])
#         metric_ts.extend(metrics['actions_dict'])
#
#     action_param_t = []
#     for metric in metric_ts:
#         # this assumes only one action is taken for one future time slot
#         if action in metric:
#             # print(metric)
#             ts = list(metric[action])[0]
#             action_param_t.append(metric[action][ts][param])
#         else:
#             action_param_t.append(None)
#
#
#     x_axis_max = len(action_param_t) / gen_len
#     x_axis = np.linspace(0, x_axis_max, len(action_param_t))
#     print(x_axis_max, x_axis)
#
#     label = action + ' ' + param
#     plt.scatter(x=x_axis, y=action_param_t, s=.5, alpha=0.3, label=label)
#     plt.title(label + ' for ' + str(gen + 1) + ' gens')
#     plt.xlabel('Generations')
#     plt.ylabel(label)
#     plt.legend()
#     plt.show()
#
#
# def plot_returns(agent_id, db_path, sim_type, gens=None, compare_with_optimal=False, smooth=0, matplotlib=True):
#     """
#
#     :param agent_id:
#     :param db_path:
#     :param sim_type:
#     :param gens:
#     :param compare_with_optimal:
#     :return:
#     """
#     # from analysis.simple_optimal_value2 import optimal_reward
#     if not gens:
#         gens = extract.gens_count(db_path, sim_type) - 1
#
#     # if compare_with_optimal:
#     #     opt_reward = optimal_reward(49, agent_id)
#     #     opt_returns = [sum(opt_reward) / 1000] * gens
#
#     returns = []
#     opt_returns = []
#     opt_gen_return = None
#
#     for gen in range(gens):
#         print(gen)
#
#         table_name = '_'.join([str(gen), sim_type, 'metrics', agent_id])
#         metrics = extract.from_metrics(db_path, gen, table_name, agent_id)
#         if 'rewards' not in metrics:
#             print('OhOh, could not find rewards ')
#             break
#         gen_return = sum(metrics['rewards']) / 1000
#         returns.append(gen_return)
#
#         if compare_with_optimal:
#             if not opt_gen_return:
#                 table_name = "0_validation_best_response-eGauge13830_metrics_eGauge13830"
#                 metrics = extract.from_metrics(db_path, gen, table_name, agent_id)
#                 opt_gen_return = sum(metrics['rewards']) / 1000
#             opt_returns.append(opt_gen_return)
#
#     returns = np.divide(returns, opt_returns)
#     opt_returns = np.divide(opt_returns, opt_returns)
#
#     if smooth:
#         smoothed_returns = []
#         for index in range(len(returns)-1):
#             average_over_entries = min(index, smooth)
#             index_start = min(index - average_over_entries, 0)
#             slice = returns[index_start:index+1]
#             smoothed_returns.append(np.mean(slice))
#
#     if not matplotlib:
#         fig = go.Figure()
#         fig.add_trace(go.Scatter(y=returns, name='achieved', mode='lines+markers'))
#         if compare_with_optimal:
#             fig.add_trace(go.Scatter(y=opt_returns, name='"optimal"', mode='lines+markers'))
#             # fig.add_trace(go.Scatter(y=returns_diff, name='difference', mode='lines+markers'))
#
#         fig.update_layout(title='Returns for ' + str(gen+1) + ' episodes',
#                           xaxis_title='Generation',
#                           yaxis_title='$')
#         fig.show()
#     else:
#         plt.rcParams.update({'font.size': 14})
#         plt.rcParams['figure.dpi'] = 300
#         plt.plot(returns, label='Return', alpha=0.7)
#         if smooth:
#             plt.plot(smoothed_returns, label='Smoothed', linewidth=2)
#
#         if compare_with_optimal:
#             plt.plot(opt_returns, label='Optimal')
#         plt.xlabel('Episodes')
#         plt.ylabel('Return [$/Episode]')
#         plt.title('Normalized and Smoothed Returns for ' + str(gen) + ' episodes')
#         plt.legend()
#         plt.grid()
#         plt.tight_layout()
#
#         plt.show()
if __name__ == "__main__":
    agent_id = 'egauge19821'
    db_path1 = 'postgresql://postgres:postgres@stargate/remote_agent_test_np_'
    db_path2 = 'postgresql://postgres:postgres@stargate/EconomicAdvantage_trade'
    db_path3 = 'postgresql://postgres:postgres@stargate/EconomicAdvantage_trade_exp'
    sim_type = 'training'

    # plot_actions(agent_id, db_path, 'csp', 'bids', 'price')
    # plot_actions(agent_id, db_path, 'validation', 'bids', 'price')
    # plot_actions(agent_id, db_path, sim_type, 'asks', 'price')
    # plot_returns(agent_id, db_path1, 'csp', compare_with_optimal=False, matplotlib=False)
    # plot_returns(agent_id, db_path3, 'validation', gens = 101, compare_with_optimal=True, smooth=1)
    # plot_returns(agent_id, db_path2, 'csp', compare_with_optimal=False, matplotlib=False)
    # plot_returns(agent_id, db_path2, 'validation', compare_with_optimal=False, matplotlib=False)
    # plot_returns(agent_id, db_path3, 'csp', compare_with_optimal=False, matplotlib=False)
    # plot_returns(agent_id, db_path3, 'validation', compare_with_optimal=False, matplotlib=False)
    # plot_metric(agent_id, db_path, 'validation', 'price_action')
    plot_metric(agent_id, db_path1, 'training', 'bid_price')
    # Todo: dissociate actions dict into its own metrics

    # plot_metric(agent_id, db_path, 'validation', 'value')
    # plot_metric(agent_id, db_path, 'csp', 'network_loss')

