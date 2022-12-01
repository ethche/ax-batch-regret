import pandas as pd
import numpy as np
import json
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Adaptive Experimentation at Scale", page_icon=":mechanical_arm:")

plt.style.use('seaborn-white')

with open("regret_df.json") as f:
    regret = json.load(f)

# dataset
regret_df = pd.DataFrame(regret)

def filter_data_line(df, reward_dist, K, T_limit):
    K_cond = df['K'] == K
    dist_cond = df['reward_dist'] == reward_dist
    T_cond = df['T'] <= T_limit

    return df[K_cond & dist_cond & T_cond]

#'Epsilon Greedy':'#377eb8',
# Things within our framework, things outside our framework...
# E.g. Dashed lines...
color_dict = {'Uniform': {"color":'#377eb8', 'ls':"--", 'hatch': '/'},
            'Batch Successive Elimination': {"color": '#984ea3', 'ls':"--", 'hatch': '/'},
            'Batch Oracle Top-Two TS': {"color": 'chocolate', 'ls':"--", 'hatch': '/'},
            'Batch Oracle TS': {"color": '#f781bf', 'ls': "--", 'hatch': '/'},
            'Gaussian Limit Top-Two TS': {"color": 'blue', 'ls': "-", 'hatch': ''},
            'Gaussian Limit TS': {"color": '#87CEEB', 'ls': "-", 'hatch': ''},
            'Myopic': {"color": '#e41a1c', 'ls': "-", 'hatch': ''},
            'Policy Gradient':  {"color": '#ff7f00','ls': "-", 'hatch': ''},
            'RHO (proposed)': {"color": '#4daf4a', 'ls': "-", 'hatch': ''}
            }

#ordered_pols = ['Uniform', ]


def plot_line(df, metric, reward_dist, K, T_limit, policies, s2, prior_type, color_dict = color_dict):

    K_cond = df['K'] == K
    dist_cond = df['reward_dist'] == reward_dist
    T_cond = df['T'] <= T_limit

    if reward_dist == "Bernoulli":
        s2 = 0.25
    
    s2_cond = df['s2'] == s2
    prior_cond = df['prior_type'] == prior_type
    policy_cond = df['policy'].isin(policies)

    plot_df = pd.DataFrame(df[K_cond & dist_cond & T_cond & policy_cond & s2_cond & prior_cond])
    plot_df['avg_regret_unif'] = plot_df['avg_regret'] * (plot_df['policy'] == 'Uniform')
    plot_df['unif_regret'] = plot_df['avg_regret_unif'].groupby(plot_df['T']).transform('sum')
    plot_df['pct_of_unif_regret'] = 100 * plot_df['avg_regret']/plot_df['unif_regret']
    plot_df['avg_correct'] = 100 * plot_df['avg_correct']

    
    if metric == "Simple Regret":
        plot_metric = "avg_regret"
    elif metric == "Percent of Simple Regret of Uniform":
        plot_metric = "pct_of_unif_regret"
    else:
        plot_metric = "avg_correct"

    fig, ax = plt.subplots()
    pivot = plot_df.pivot('T', 'policy', plot_metric)

    ordered_pols = [key for key in color_dict.keys() if key in pivot.columns]
    pivot = pivot[ordered_pols]

    pivot.plot(color = [color_dict.get(x)['color'] for x in pivot.columns], 
                style = [color_dict.get(x)['ls'] for x in pivot.columns], 
                linewidth = 6, figsize = (12,9), ax=ax)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=20)
    #ax.legend(bbox_to_anchor=(0.5, -0.1), loc="upper right", fontsize=20)
    #ax.legend(fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_ylabel(metric, fontsize=20)
    ax.set_xlabel("Number of reallocations", fontsize=20)
    ax.set_xticks(pivot.index, fontsize = 20)
    st.pyplot(fig)

@st.cache
def filter_bar_s2(df, metric, K, T, policies, prior_type):

    K_cond = df['K'] == K
    T_cond = df['T'] == T
    prior_cond = df['prior_type'] == prior_type
    policy_cond = df['policy'].isin(policies)
    reward_cond = df['reward_dist'] == "Gumbel"

    plot_df = pd.DataFrame(df[K_cond & T_cond & reward_cond & policy_cond & prior_cond])
    plot_df['avg_regret_unif'] = plot_df['avg_regret'] * (plot_df['policy'] == 'Uniform')
    plot_df['unif_regret'] = plot_df['avg_regret_unif'].groupby(plot_df['s2']).transform('sum')
    plot_df['pct_of_unif_regret'] = 100 * plot_df['avg_regret']/plot_df['unif_regret']
    plot_df['avg_correct'] = 100 * plot_df['avg_correct']
    
    if metric == "Simple Regret":
        plot_metric = "avg_regret"
    elif metric == "Percent of Simple Regret of Uniform":
        plot_metric = "pct_of_unif_regret"
    else:
        plot_metric = "avg_correct"

    pivot = plot_df.pivot('s2', 'policy', plot_metric)
    return pivot

@st.cache
def filter_bar_prior(df, metric, reward_dist, K, T, policies, s2, color_dict = color_dict):

    K_cond = df['K'] == K
    T_cond = df['T'] == T
    dist_cond = df['reward_dist'] == reward_dist
    if reward_dist == "Bernoulli":
        s2 = 0.25
    s2_cond = df['s2'] == s2
    policy_cond = df['policy'].isin(policies)

    plot_df = pd.DataFrame(df[dist_cond & K_cond & T_cond & policy_cond & s2_cond])
    plot_df['avg_regret_unif'] = plot_df['avg_regret'] * (plot_df['policy'] == 'Uniform')
    plot_df['unif_regret'] = plot_df['avg_regret_unif'].groupby(plot_df['prior_type']).transform('sum')
    plot_df['pct_of_unif_regret'] = 100 * plot_df['avg_regret']/plot_df['unif_regret']
    plot_df['avg_correct'] = 100 * plot_df['avg_correct']
    
    if metric == "Simple Regret":
        plot_metric = "avg_regret"
    elif metric == "Percent of Simple Regret of Uniform":
        plot_metric = "pct_of_unif_regret"
    else:
        plot_metric = "avg_correct"

    fig, ax = plt.subplots()
    pivot = plot_df.pivot('prior_type', 'policy', plot_metric)
    pivot = pivot.reindex(["Flat", "Top One", "Top Half", "Descending"])

    return pivot

def plot_bar(pivot, metric, bar_x, color_dict = color_dict):
    fig, ax = plt.subplots()
    ordered_pols = [key for key in color_dict.keys() if key in pivot.columns]
    pivot = pivot[ordered_pols]

    pivot.plot.bar(color = [color_dict.get(x)['color'] for x in pivot.columns], linewidth = 4, figsize = (12,8), ax=ax)
    bars = ax.patches
    hatch_styles = ''.join([color_dict[x]['hatch'] for x in pivot.columns])
    hatches = ''.join(h*len(pivot) for h in hatch_styles)

    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=20)
    #ax.legend(fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_ylabel(metric, fontsize=20)
    ax.set_xlabel(bar_x, fontsize=20)

    if metric == "Percent of Simple Regret of Uniform":
        ax.set_ylim((20,105))

    
    #ax.set_xticks(pivot.index, fontsize = 18)
    st.pyplot(fig)



# Top line
title_container = st.container()

with title_container:
    st.title("Adaptive Experimentation at Scale: Bayesian Batch Policies")

# with row1_2:
#     st.write(
#         """
#     ###
#     We compare the performance of adaptive experimentation policies in batch environments.
#     The focal metric is **Bayes simple regret**: the difference in the average reward between the best arm
#     and the arm chosen by the experimenter at the end of the experiment, averaged over the prior.
#     """
#     )

row1_sep_1, row1_sep_2 = st.columns((5,1))

with row1_sep_1:
    st.write("""
    ## Overview

    We study a batched adaptive experiment: at each epoch, we assign treatment arms, observe corresponding rewards, 
    and use them to reallocate sampling effort in the next batch.
    The goal is to select the arm with the highest reward at the end of the experiment (best arm identification). 
    We benchmark the performance of a wide range of batch adaptive experimentation policies.

    ## Methods
    """)

gauss, oracle, freq = st.tabs(["Gaussian Approximation", "Oracle Policies", "Frequentist Policies"])

with gauss:
    st.write("""
    Sampling policies based on Gaussian sequential experiment, the Gaussian approximation of aggregated rewards.
    - **Residual Horizon Optimization (proposed)**: selects the sampling allocation by solving the RHO planning problem.  
    - **Policy Gradient**: allocates samples according to a policy parameterized by a feed-forward NN, which is trained through policy gradient with an episode length of 5 batches.    
    - **Myopic:** selects the sampling allocation that maximizes the one-step lookahead Q-function. A randomized version of Knowledge Gradient.  
    - **Gaussian Limit TS**: Thompson Sampling policy for the Gaussian sequential experiment.  
    - **Gaussian Limit Top-Two TS**: Top-Two Thompson Sampling policy for the Gaussian sequential experiment.  
    """)
    
with oracle:
    st.write("""
    Oracle policies that assume true knowledge of the distribution of individual rewards. Only available when raw rewards follow a Beta-Bernoulli model.
    - **Oracle TS**: Beta-Bernoulli Thompson Sampling policy with batch updates.
    - **Oracle Top-Two TS**: Beta-Bernoulli Top-Two Thompson Sampling policy with batch updates.
    """)

with freq:
    st.write("""
    Standard frequentist policies.
    - **Uniform:** uniformly samples all arms in every batch.  
    - **Batch Successive Elimination:** in every time period, arm is eliminated if its upper confidence bound is below the lower confidence bound of another arm. 
        - If an arm $a$ is sampled $n_{a}$ times, then its confidence bound is: $C_{a} = c \cdot s_{a}\sqrt{\\frac{\log(Kn_{a}^{2}/\delta))}{n_{a}}}$,
        - $s_{a}$ is measurement standard deviation, $K$ is the number of arms, $c$ and $\delta$ are chosen optimally for each instance via grid search.
    """)

setting, space = st.columns((5,1))

with setting:
    st.write("""
    ## Setting

    We evaluate methods using the Bayes simple regret, the optimality gap between the selected arm and the best arm, 
    averaged across instances drawn from the experimenter's prior. We propose Bayesian policies that use 
    Gaussian approximations of the aggregated rewards in each batch in order to update beliefs about the average rewards. 
    """)

gumbel, bernoulli  = st.tabs(["Gumbel", "Bernoulli"])

with bernoulli:
    st.write("""
    **Bernoulli/Beta**: 
    - Treatment rewards for each arm $a$ are distributed as $Bernoulli(\\theta_{a})$ r.v.s. 
    - The experimenter has an independent Beta(100,100) prior over each $\\theta_{a}$. 
    - The measurement variance is fixed by $\\theta$, so this cannot be altered.
    """)

with gumbel:
    st.write("""
    **Gumbel/Gamma**: 
    - Treatment rewards for each arm $a$ are distributed as $Gumbel(\mu_{a}, \\beta)$ r.v.s. 
    - The experimenter has an independent Gamma(100, 1/100) prior over each $\mu_{a}$. 
    - The $\\beta$ parameter is known and determines the measurement variance $s_{a}^{2}$. We consider measurement variances in $(0.2, 1, 5)$.
    """)

priors, space = st.columns((5,1))

with priors:
    st.write("""
    ## Priors

    We consider different prior distributions on the average rewards. 
    Unlike Bayesian bandit methods such as Thompson sampling, we do not require a model of the overall reward distribution.


    """)

flat, top_one, top_half, descending = st.tabs(['Flat', 'Top One', 'Top Half', 'Descending'])

with flat:
    st.write("Below displays the mean $\pm$ one standard deviation under each prior distribution type (for $K = 6$ treatment arms).")
    st.image("./fig/flat.jpg")
    st.text("")

with top_one:
    st.write("Below displays the mean $\pm$ one standard deviation under each prior distribution type (for $K = 6$ treatment arms).")
    st.image("./fig/top_one.jpg")
    st.text("")

with top_half:
    st.write("Below displays the mean $\pm$ one standard deviation under each prior distribution type (for $K = 6$ treatment arms).")
    st.image("./fig/top_half.jpg")
    st.text("")

with descending:
    st.write("Below displays the mean $\pm$ one standard deviation under each prior distribution type (for $K = 6$ treatment arms).")
    st.image("./fig/descending.jpg")
    st.text("")

row1_header_1, space, row1_header_2 = st.columns((3, 0.2, 1.5))

with row1_header_1:
    st.write("""
    ## Results
    ### Simple Regret across Reallocation Epochs
    """
    )

# line_plt, space = st.columns((5, 0.2))
# row2_1, space, row2_2, space= st.columns((3,0.2,3,0.2))

# with row2_1:
#     st.text("")
#     st.text("")
#     dist_selected = st.selectbox("Reward Distribution", options = ('Gumbel', 'Bernoulli'), key = 'dist_select')
#     metric_selected = st.selectbox("Metric", options = ('Percent of Simple Regret of Uniform', 'Simple Regret', 'Percent Correct'), 
#                                     key = 'metric_select')
#     K_selected = st.selectbox("Number of Treatment Arms", options = (10, 100), key = 'K_select')
#     s2_selected = st.selectbox("Measurement variance", options = (1, 0.2, 5), key = 's2_select')

# with row2_2:
#     st.text("")
#     st.text("")
    
#     policies_selected = st.multiselect("Policies", options = [key for key in  color_dict.keys()], key = 'policies_select', 
#                                         default = ['Uniform', 
#                                                    'Batch Successive Elimination', 
#                                                    'Gaussian Limit TS', 
#                                                    'Batch Oracle TS',
#                                                    'Myopic',
#                                                    'Policy Gradient',
#                                                    'RHO (proposed)'])
#     prior_type_selected = st.selectbox("Prior Distribution", options = ('Flat', 'Top One', 'Top Half', 'Descending'), 
#                                         key = 'prior_select')
#     T_selected = st.select_slider("Max Horizon", options = [i for i in range(1,11)], value = (1,10), key = 'T_select')

line_plt, space, line_plt_toggle = st.columns((3, 0.2, 1.5))

with line_plt_toggle:
    st.text("")
    st.text("")
    dist_selected = st.radio("Reward Distribution", options = ('Gumbel', 'Bernoulli'), key = 'dist_select')

    K_selected = st.radio("Number of Treatment Arms", options = (10, 100), key = 'K_select')
    metric_selected = st.selectbox("Metric", options = ('Percent of Simple Regret of Uniform', 'Simple Regret', 'Percent Correct'), 
                                    key = 'metric_select')
    s2_selected = st.selectbox("Measurement variance", options = (1, 0.2, 5), key = 's2_select')

    policies_selected = st.multiselect("Policies", options = [key for key in  color_dict.keys()], key = 'policies_select', 
                                        default = ['Uniform', 
                                                   'Batch Successive Elimination', 
                                                   'Gaussian Limit TS', 
                                                   'Batch Oracle TS',
                                                   'Myopic',
                                                   'Policy Gradient',
                                                   'RHO (proposed)'])
    prior_type_selected = st.radio("Prior Distribution", options = ('Flat', 'Top One', 'Top Half', 'Descending'), 
                                        key = 'prior_select')
    T_selected = st.select_slider("Max Horizon", options = [i for i in range(1,11)], value = (1,10), key = 'T_select')

with line_plt:
    st.text("")
    st.text("")
    plot_line(regret_df, 
              metric_selected, 
              dist_selected, 
              K_selected, 
              T_selected[1], 
              policies_selected, 
              s2_selected, 
              prior_type_selected)




row2_sep_1, space, row2_sep_2 = st.columns((3, 0.2, 1.5))

with row2_sep_1:
    st.write("""
    ### Simple Regret across Measurement Variances
    """
    )

row3_1, space, row3_2 = st.columns((3, 0.2, 1.5))

with row3_2:
    st.text("")
    bar_reward_selected = st.radio("Reward Distribution", options = ['Gumbel'], 
                                    key = 'bar_reward_select')

    bar_K_selected = st.radio("Number of Treatment Arms", options = (10, 100), key = 'bar_K_select')

    bar_metric_selected = st.selectbox("Metric", options = ('Percent of Simple Regret of Uniform', 'Simple Regret', 'Percent Correct'), 
                                    key = 'bar_metric_select')
    bar_policies_selected = st.multiselect("Policies", options = [key for key in  color_dict.keys()], key = 'bar_policies_select', 
                                        default = ['Uniform', 
                                                   'Batch Successive Elimination', 
                                                   'Gaussian Limit TS', 
                                                   'Batch Oracle TS',
                                                   'Myopic',
                                                   'Policy Gradient',
                                                   'RHO (proposed)'])
    bar_prior_type_selected = st.radio("Prior Distribution", options = ('Flat', 'Top One', 'Top Half', 'Descending'), 
                                        key = 'bar_prior_select')
    bar_T_selected = st.select_slider("T", options = [i for i in range(1,11)], value = (1,10), key = 'bar_T_select')
    st.text("")

with row3_1:
    st.text("")
    plot_bar(filter_bar_s2(regret_df, 
                            bar_metric_selected, 
                            bar_K_selected, 
                            bar_T_selected[1], 
                            bar_policies_selected,
                            bar_prior_type_selected), bar_metric_selected, "Measurement Variance")


    st.text("")

row3_sep_1, space, row3_sep_2 = st.columns((3, 0.2, 1.5))

with row3_sep_1:
    st.write("""
    ### Simple Regret across Priors
    """
    )

row4_1, space, row4_2 = st.columns((3, 0.2, 1.5))

with row4_2:
    st.text("")
    prior_dist_selected = st.radio("Reward Distribution", options = ('Gumbel', 'Bernoulli'), key = 'prior_dist_select')
    prior_K_selected = st.radio("Number of Treatment Arms", options = (10, 100), key = 'prior_K_select')
    prior_metric_selected = st.selectbox("Metric", options = ('Percent of Simple Regret of Uniform', 'Simple Regret', 'Percent Correct'), 
                                    key = 'prior_metric_select')     
    prior_s2_selected = st.selectbox("Measurement variance", options = (1, 0.2, 5), key = 'prior_s2_select')
    prior_policies_selected = st.multiselect("Policies", options = [key for key in  color_dict.keys()], key = 'prior_policies_select', 
                                        default = ['Uniform', 
                                                   'Batch Successive Elimination', 
                                                   'Gaussian Limit TS', 
                                                   'Batch Oracle TS',
                                                   'Myopic',
                                                   'Policy Gradient',
                                                   'RHO (proposed)'])
    prior_T_selected = st.select_slider("T", options = [i for i in range(1,11)], value = (1,10), key = 'prior_T_select')

with row4_1:
    st.text("")
    plot_bar(filter_bar_prior(regret_df,
                prior_metric_selected, 
                prior_dist_selected,
                prior_K_selected, 
                prior_T_selected[1], 
                prior_policies_selected,
                prior_s2_selected), prior_metric_selected, "Prior Distribution")
