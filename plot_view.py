import pandas as pd
import numpy as np
import json
import streamlit as st
import matplotlib.pyplot as plt



st.set_page_config(layout="wide")

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
color_dict = {'Uniform': '#377eb8',
            'Successive Elimination': '#4daf4a',
            'Bern-TTTS': 'brown',
            'Bern-TS': 'brown',
            'Batch-Limit-TTTS': 'blue',
            'Batch-Limit-TS':'blue',
            'KG': 'red',
            'PG-5 (ours)': '#ff7f00', 
            'Q-myopic (ours)': '#984ea3'
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

    pivot.plot(color = [color_dict.get(x) for x in pivot.columns], linewidth = 5, figsize = (12,8), ax=ax)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=20)
    #ax.legend(fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_ylabel(metric, fontsize=20)
    ax.set_xlabel("Number of reallocations", fontsize=20)
    ax.set_xticks(pivot.index, fontsize = 20)
    st.pyplot(fig)

def plot_bar_s2(df, metric, K, T, policies, prior_type, color_dict = color_dict):

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

    fig, ax = plt.subplots()
    pivot = plot_df.pivot('s2', 'policy', plot_metric)

    ordered_pols = [key for key in color_dict.keys() if key in pivot.columns]
    pivot = pivot[ordered_pols]

    pivot.plot.bar(color = [color_dict.get(x) for x in pivot.columns], linewidth = 4, figsize = (12,8), ax=ax)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=20)
    #ax.legend(fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_ylabel(metric, fontsize=20)
    ax.set_xlabel("Measurement Variance", fontsize=20)
    #ax.set_xticks(pivot.index, fontsize = 18)
    st.pyplot(fig)

def plot_bar_prior(df, metric, reward_dist, K, T, policies, s2, color_dict = color_dict):

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

    ordered_pols = [key for key in color_dict.keys() if key in pivot.columns]
    pivot = pivot[ordered_pols]

    pivot.plot.bar(color = [color_dict.get(x) for x in pivot.columns], linewidth = 4, figsize = (12,8), ax=ax)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=20)
    #ax.legend(fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_ylabel(metric, fontsize=20)
    ax.set_xlabel("Prior Distribution", fontsize=20)
    #ax.set_xticks(pivot.index, fontsize = 18)
    st.pyplot(fig)

# Top line
row1_1, space, row1_2 = st.columns((3, 0.2, 1.75))

with row1_1:
    st.title("Regret of Batch Experimentation Policies")

# with row1_2:
#     st.write(
#         """
#     ###
#     We compare the performance of adaptive experimentation policies in batch environments.
#     The focal metric is **Bayes simple regret**: the difference in the average reward between the best arm
#     and the arm chosen by the experimenter at the end of the experiment, averaged over the prior.
#     """
#     )

row1_sep_1, space, row1_sep_2 = st.columns((3, 0.2, 1.75))

with row1_sep_1:
    st.write("""
    ## Simple Regret across Reallocations
    """
    )

row2_1, space, row2_2 = st.columns((3, 0.2, 1.75))

with row2_2:
    st.text("")
    st.text("")
    dist_selected = st.selectbox("Reward Distribution", options = ('Gumbel', 'Bernoulli'), key = 'dist_select')
    metric_selected = st.selectbox("Metric", options = ('Percent of Simple Regret of Uniform', 'Simple Regret', 'Percent Correct'), 
                                    key = 'metric_select')
    K_selected = st.selectbox("Number of Treatment Arms", options = (10, 100), key = 'K_select')
    s2_selected = st.selectbox("Measurement variance", options = (1, 0.2, 5), key = 's2_select')
    policies_selected = st.multiselect("Policies", options = [key for key in  color_dict.keys()], key = 'policies_select', 
                                        default = ['Uniform', 
                                                   'Successive Elimination', 
                                                   'Batch-Limit-TS', 
                                                   'Bern-TS',
                                                   'KG',
                                                   'PG-5 (ours)',
                                                   'Q-myopic (ours)'])
    prior_type_selected = st.selectbox("Prior Distribution", options = ('Flat', 'Top One', 'Top Half', 'Descending'), 
                                        key = 'prior_select')
    T_selected = st.select_slider("Max Horizon", options = [i for i in range(1,11)], value = (1,10), key = 'T_select')



with row2_1:
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


    st.write("""

    ##### **Notes:**  
    - Each batch has $n = 100$ samples, which are allocated across the treatment arms in every time period.
    - The experimenter has a prior over the means of the arm rewards. 
    - We consider two types of reward distributions:
        1. **Gumbel**: rewards follow independent $Gumbel(\mu , \\beta)$ distributions with a $Gamma(100, 1/100)$ prior over $\mu$. The $\\beta$ parameter is known and determines the measurement variance. We consider measurement variances $\in [1/5, 1, 5]$.
        2. **Bernoulli**: rewards follow independent $Bernoulli(\\theta)$ distributions with a $Beta(100,100)$ prior over $\\theta$. The measurement variance is fixed by $\\theta$, so this cannot be altered.
    - For policies that use the Gaussian batch approximations, the true prior is also approximated by a Gaussian distribution with the same mean and variance to preserve conjugacy.
    - We consider different prior distributions. Below displays the mean and standard deviations under each prior distribution type.
    """)
    

    

prior_img_1, prior_img_2 = st.columns((3, 3))

with prior_img_1:
    st.image("./prior_fig_1.jpg")
    st.text("")

with prior_img_2:
    st.image("./prior_fig_2.jpg")
    st.text("")

row2_sep_1, space, row2_sep_2 = st.columns((3, 0.2, 1.75))

with row2_sep_1:
    st.write("""
    ## Simple Regret across Measurement Variances
    """
    )

row3_1, space, row3_2 = st.columns((3, 0.2, 1.75))

with row3_2:
    st.text("")
    bar_metric_selected = st.selectbox("Metric", options = ('Percent of Simple Regret of Uniform', 'Simple Regret', 'Percent Correct'), 
                                    key = 'bar_metric_select')
    bar_K_selected = st.selectbox("Number of Treatment Arms", options = (10, 100), key = 'bar_K_select')
    bar_policies_selected = st.multiselect("Policies", options = [key for key in  color_dict.keys()], key = 'bar_policies_select', 
                                        default = ['Uniform', 
                                                   'Successive Elimination', 
                                                   'Batch-Limit-TS', 
                                                   'Bern-TS',
                                                   'KG',
                                                   'PG-5 (ours)',
                                                   'Q-myopic (ours)'])
    bar_prior_type_selected = st.selectbox("Prior Distribution", options = ('Flat', 'Top One', 'Top Half', 'Descending'), 
                                        key = 'bar_prior_select')
    bar_T_selected = st.select_slider("T", options = [i for i in range(1,11)], value = (1,10), key = 'bar_T_select')
    st.text("")

with row3_1:
    st.text("")
    plot_bar_s2(regret_df, 
                bar_metric_selected, 
                bar_K_selected, 
                bar_T_selected[1], 
                bar_policies_selected,
                bar_prior_type_selected)

    st.write("""

    ##### **Notes:**  
    - We only display the results for the Gumbel specification, since for Bernoulli rewards, the reward means and measurement noise are fixed by the same parameter.
        """)

    st.text("")

row3_sep_1, space, row3_sep_2 = st.columns((3, 0.2, 1.75))

with row3_sep_1:
    st.write("""
    ## Simple Regret across Priors
    """
    )

row4_1, space, row4_2 = st.columns((3, 0.2, 1.75))

with row4_2:
    st.text("")
    prior_dist_selected = st.selectbox("Reward Distribution", options = ('Gumbel', 'Bernoulli'), key = 'prior_dist_select')
    prior_metric_selected = st.selectbox("Metric", options = ('Percent of Simple Regret of Uniform', 'Simple Regret', 'Percent Correct'), 
                                    key = 'prior_metric_select')                           
    prior_K_selected = st.selectbox("Number of Treatment Arms", options = (10, 100), key = 'prior_K_select')
    prior_s2_selected = st.selectbox("Measurement variance", options = (1, 0.2, 5), key = 'prior_s2_select')
    prior_policies_selected = st.multiselect("Policies", options = [key for key in  color_dict.keys()], key = 'prior_policies_select', 
                                        default = ['Uniform', 
                                                   'Successive Elimination', 
                                                   'Batch-Limit-TS', 
                                                   'Bern-TS',
                                                   'KG',
                                                   'PG-5 (ours)',
                                                   'Q-myopic (ours)'])
    prior_T_selected = st.select_slider("T", options = [i for i in range(1,11)], value = (1,10), key = 'prior_T_select')

with row4_1:
    st.text("")
    plot_bar_prior(regret_df,
                prior_metric_selected, 
                prior_dist_selected,
                prior_K_selected, 
                prior_T_selected[1], 
                prior_policies_selected,
                prior_s2_selected)

row5_1, space, row5_2 = st.columns((3, 0.2, 1.75))

with row5_1:

    st.write(
        """
        ## Policies
        - **Uniform:** uniformly samples all arms in every batch.  
        - **Successive Elimination:** in every time period, arm is eliminated if its upper confidence bound is below the lower confidence bound of another arm.  
        - **Batch-Limit TS$^{*}$ / Top-Two-TS$^{*}$:** allocates samples according to Thompson Sampling / Top-Two Thompson Sampling probabilities under Gaussian batch approximations.  
        - **Bern TS / Top-Two-TS$:** allocates samples according to Thompson Sampling / Top-Two Thompson Sampling probabilities for Bernoulli rewards. Only available for Bernoulli specification.  
        - **KG$^{*}$:** allocates samples to maximize the one-step lookahead Q-function. A randomized version of Knowledge Gradient.  
        - **PG-5$^{*}$ (ours)**: allocates samples according to policy trained by policy gradient with an episode length of 5 batches.  
        - **Q-myopic$^{*}$ (ours)**: allocates samples to maximize the Q-myopic planning problem.  
        
        $^{*}$ denotes policies that use Gaussian batch approximations as well as Gaussian approximations of the prior distribution.
        """
    )
        
