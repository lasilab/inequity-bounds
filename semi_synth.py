from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from folktables import ACSDataSource, ACSEmployment


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
np.random.seed(0)

from utils import compute_plugin_bounds, compute_bias_corrected_bounds, compute_gamma_plugin_bounds, compute_gamma_bias_corrected_bounds
from utils import compute_lower_gamma_nonsmooth, compute_upper_gamma_nonsmooth

def simulate_outcome_treatment(x, group):
    '''
    Function to simulate outcome and treatment
    '''

    sum_x = np.sum(x, axis=1)

    probs_y0 = 1 / (1 + np.exp(-sum_x - 1))
    y0 = np.random.binomial(1, probs_y0)
    y1 = np.random.binomial(1, probs_y0 / 2)

    probs_treatment = 1 / (1 + np.exp(-sum_x + 1.5))    
    # make treatment more likely for group 1
    probs_treatment[np.where(group == 1)] = (1 / (1 + np.exp(-sum_x + 2.75)))[np.where(group == 1)] 
    probs_treatment[np.where(y0 == 0)] = probs_treatment[np.where(y0 == 0)] / 1.5 # unobserved confounding in treatment with gamma = 1.5
    
    treatment = np.random.binomial(1, probs_treatment)
    y = treatment * y1 + (1 - treatment) * y0
    return treatment, y, y0, y1

def get_folktables_data():
    '''
    Function to generate semi-synthetic data from Folktables (US Census data)
    '''

    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=["CA"], download=True) # download from CA 
    features, _, group = ACSEmployment.df_to_numpy(acs_data)

    # filter to only contain data with group == 1 (white) or 2 (black)
    indices = np.where((group == 1) | (group == 2))
    features = features[indices]
    group = group[indices]

    # denote group 0 as white and group 1 as black
    group = group - 1
    var_target = 0.05

    # add group to features
    features = np.concatenate((features, group.reshape(-1, 1)), axis=1)
    scaler = StandardScaler()
    scaler.fit(features)
    features = scaler.transform(features)
    features *= var_target

    # split into two sets of pre and post
    np.random.seed = 0
    indices = np.arange(len(features))
    np.random.shuffle(indices)

    pre_inds = indices[:int(len(features)/2)]
    post_inds = indices[int(len(features)/2):]

    pre_features = features[pre_inds]
    pre_group = group[pre_inds]

    post_features = features[post_inds]
    post_group = group[post_inds]

    # do k fold splitting on pre and post
    kf = KFold(n_splits=5)
    pre_train_test_sets = []

    for train_index, test_index in kf.split(pre_features):

        train_features = pre_features[train_index]
        train_group = pre_group[train_index]
        train_treatment, train_y, train_y0, train_y1 = simulate_outcome_treatment(train_features, train_group)
        # for pre treatment data, train_treatment is always 0, train_y is always train_y0
        train_treatment = np.zeros(len(train_treatment))
        train_y = train_y0

        # process test data
        test_features = pre_features[test_index]
        test_group = pre_group[test_index]
        test_treatment, test_y, test_y0, test_y1 = simulate_outcome_treatment(test_features, test_group)
        # for pre treatment data, test_treatment is always 0, test_y is always test_y0
        test_treatment = np.zeros(len(test_treatment))
        test_y = test_y0
        pre_train_test_sets.append((train_features, train_treatment, train_y, train_y0, train_y1, train_group, test_features, test_treatment, test_y, test_y0, test_y1, test_group))   
    
    post_train_test_sets = []
    for train_index, test_index in kf.split(post_features):
        train_features = post_features[train_index]
        train_group = post_group[train_index] 
        train_treatment, train_y, train_y0, train_y1 = simulate_outcome_treatment(train_features, train_group)

        test_features = post_features[test_index]
        test_group = post_group[test_index]
        test_treatment, test_y, test_y0, test_y1 = simulate_outcome_treatment(test_features, test_group)

        post_train_test_sets.append((train_features, train_treatment, train_y, train_y0, train_y1, train_group, test_features, test_treatment, test_y, test_y0, test_y1, test_group))
    return pre_train_test_sets, post_train_test_sets

if __name__ == "__main__":

    pre_train_test_sets, post_train_test_sets = get_folktables_data()

    avg_values_0 = []
    avg_values_1 = []
    avg_true_values_0 = []
    avg_true_values_1 = []
    gammas = np.arange(1.0, 2.1, 0.1)

    total_ci_estimates_l0 = []
    total_ci_estimates_u0 = []

    total_ci_estimates_l1 = []
    total_ci_estimates_u1 = []

    for i in range(5): # splits
        pre_train_features, pre_train_treatment, pre_train_y, pre_train_y0, pre_train_y1, pre_train_group, pre_test_features, pre_test_treatment, pre_test_y, pre_test_y0, pre_test_y1, pre_test_group = pre_train_test_sets[i]
        post_train_features, post_train_treatment, post_train_y, post_train_y0, post_train_y1, post_train_group, post_test_features, post_test_treatment, post_test_y, post_test_y0, post_test_y1, post_test_group = post_train_test_sets[i]

        # compute P(treatment = 1 | group = 0, y(0) = 1, D = 1) on post
        inds0 = np.where((post_test_group == 0) & (post_test_y0 == 1))
        inds1 = np.where((post_test_group == 1) & (post_test_y0 == 1))

        # compute true rate of treatment among needy for group 0 and group 1
        true_rate_0 = np.mean(post_test_treatment[inds0])
        true_rate_1 = np.mean(post_test_treatment[inds1])

        avg_true_values_0.append(true_rate_0)
        avg_true_values_1.append(true_rate_1)

        mu = LogisticRegression(penalty=None, solver='lbfgs')
        mu.fit(pre_train_features, pre_train_y)
        print("mu acuracy", accuracy_score(pre_test_y, mu.predict(pre_test_features)))

        pi = LogisticRegression(penalty=None, solver='lbfgs')
        pi.fit(post_train_features, post_train_treatment)
        print("pi acuracy", accuracy_score(post_test_treatment, pi.predict(post_test_features)))

        # compute P(Y = 1, D=0)
        num_0 = np.sum(pre_train_y[np.where(pre_train_group == 0)])
        num_1 = np.sum(pre_train_y[np.where(pre_train_group == 1)])
        denom_0 = len(pre_train_y[np.where(pre_train_group == 0)]) + len(post_train_y[np.where(post_train_group == 0)])
        denom_1 = len(pre_train_y[np.where(pre_train_group == 1)]) + len(post_train_y[np.where(post_train_group == 1)])
        pg_0 = num_0 / denom_0
        pg_1 = num_1 / denom_1

        # combine pre and post test data
        test_features = np.concatenate((post_test_features, pre_test_features))
        test_treatment = np.concatenate((post_test_treatment, pre_test_treatment))
        test_y = np.concatenate((post_test_y, pre_test_y))
        test_group = np.concatenate((post_test_group, pre_test_group))
        test_D = np.concatenate((np.ones(len(post_test_features)), np.zeros(len(pre_test_features))))

        # split into group 0 and group 1
        test_features_0 = test_features[np.where(test_group == 0)]
        test_treatment_0 = test_treatment[np.where(test_group == 0)]
        test_y_0 = test_y[np.where(test_group == 0)]
        test_D_0 = test_D[np.where(test_group == 0)]
        
        test_features_1 = test_features[np.where(test_group == 1)]
        test_treatment_1 = test_treatment[np.where(test_group == 1)]
        test_y_1 = test_y[np.where(test_group == 1)]
        test_D_1 = test_D[np.where(test_group == 1)]

        # indicators
        indicator_d0_0 = (test_D_0 == 0)
        indicator_d0_1 = (test_D_1 == 0)
        indicator_d1_0 = (test_D_0 == 1)
        indicator_d1_1 = (test_D_1 == 1)

        # run classifiers
        mu_probs_group0 = mu.predict_proba(test_features_0)[:, 1]
        mu_probs_group1 = mu.predict_proba(test_features_1)[:, 1]

        pi_probs_group0 = pi.predict_proba(test_features_0)[:, 1]
        pi_probs_group1 = pi.predict_proba(test_features_1)[:, 1]

        g_probs_group0, g_probs_group1 = 0.5, 0.5
        g_over_1_minus_g_0 = g_probs_group0 / (1 - g_probs_group0)
        g_over_1_minus_g_1 = g_probs_group1 / (1 - g_probs_group1)

        # compute plugin
        pid_lower_plugin0, pid_upper_plugin0 = compute_plugin_bounds(g_probs_group0, pi_probs_group0, mu_probs_group0, pg_0)
        pid_lower_plugin1, pid_upper_plugin1 = compute_plugin_bounds(g_probs_group1, pi_probs_group1, mu_probs_group1, pg_1)

        # compute bias corrected
        pid_lower_bc0, pid_upper_bc0 = compute_bias_corrected_bounds(g_probs_group0, pi_probs_group0, mu_probs_group0, pg_0, indicator_d0_0, indicator_d1_0, test_treatment_0, test_y_0)
        pid_lower_bc1, pid_upper_bc1 = compute_bias_corrected_bounds(g_probs_group1, pi_probs_group1, mu_probs_group1, pg_1, indicator_d0_1, indicator_d1_1, test_treatment_1, test_y_1)

        u_vals0, l_vals0 = [], []
        u_vals1, l_vals1 = [], []

        u_vals0_combined, l_vals0_combined = [], []
        u_vals1_combined, l_vals1_combined = [], []

        for gamma in gammas:
            gamma_prime = 1 / gamma

            ###### PLUG IN BOUNDS ######
            lower_plugin_gamma_0, upper_plugin_gamma_0 = compute_gamma_plugin_bounds(gamma, gamma_prime, mu_probs_group0, pi_probs_group0, pg_0, g_probs_group0)
            lower_plugin_gamma_1, upper_plugin_gamma_1 = compute_gamma_plugin_bounds(gamma, gamma_prime, mu_probs_group1, pi_probs_group1, pg_1, g_probs_group1)

            ##### Bias Corrected Estimates of Bounds
            lower_bc_gamma_0, upper_bc_gamma_0 = compute_gamma_bias_corrected_bounds(gamma, gamma_prime, mu_probs_group0, pi_probs_group0, pg_0, g_over_1_minus_g_0, indicator_d0_0, indicator_d1_0, test_treatment_0, test_y_0)
            lower_bc_gamma_1, upper_bc_gamma_1 = compute_gamma_bias_corrected_bounds(gamma, gamma_prime, mu_probs_group1, pi_probs_group1, pg_1, g_over_1_minus_g_1, indicator_d0_1, indicator_d1_1, test_treatment_1, test_y_1)

            # performing min and max operations
            upper_bound_gamma_0 = compute_upper_gamma_nonsmooth(upper_plugin_gamma_0, upper_bc_gamma_0)
            upper_bound_gamma_1 = compute_upper_gamma_nonsmooth(upper_plugin_gamma_1, upper_bc_gamma_1)
            lower_bound_gamma_0 = compute_lower_gamma_nonsmooth(pid_lower_plugin0, pid_lower_bc0, lower_plugin_gamma_0, lower_bc_gamma_0)
            lower_bound_gamma_1 = compute_lower_gamma_nonsmooth(pid_lower_plugin1, pid_lower_bc1, lower_plugin_gamma_1, lower_bc_gamma_1)

            # compute average values
            upper_bound_gamma_0_avg = np.mean(upper_bound_gamma_0)
            lower_bound_gamma_0_avg = np.mean(lower_bound_gamma_0)
            upper_bound_gamma_1_avg = np.mean(upper_bound_gamma_1)
            lower_bound_gamma_1_avg = np.mean(lower_bound_gamma_1)

            u_vals0.append(upper_bound_gamma_0_avg)
            l_vals0.append(lower_bound_gamma_0_avg)
            u_vals1.append(upper_bound_gamma_1_avg)
            l_vals1.append(lower_bound_gamma_1_avg)

            # compute combined bounds
            u_vals0_combined.append(upper_bound_gamma_0)
            l_vals0_combined.append(lower_bound_gamma_0)
            u_vals1_combined.append(upper_bound_gamma_1)
            l_vals1_combined.append(lower_bound_gamma_1)

        avg_values_0.append((l_vals0, u_vals0))
        avg_values_1.append((l_vals1, u_vals1))

        l_vals0_combined = np.array(l_vals0_combined)
        u_vals0_combined = np.array(u_vals0_combined)
        l_vals1_combined = np.array(l_vals1_combined)
        u_vals1_combined = np.array(u_vals1_combined)

        total_ci_estimates_l0.append(l_vals0_combined)
        total_ci_estimates_u0.append(u_vals0_combined)
        total_ci_estimates_l1.append(l_vals1_combined)
        total_ci_estimates_u1.append(u_vals1_combined)

    # convert to np arrays
    avg_values_0 = np.array(avg_values_0)
    avg_values_1 = np.array(avg_values_1)

    # convert list of (11, 46504) to (11, 5 * 46504)
    total_ci_estimates_l0 = np.concatenate(total_ci_estimates_l0, axis=1)
    total_ci_estimates_u0 = np.concatenate(total_ci_estimates_u0, axis=1)

    total_ci_estimates_l1 = np.concatenate(total_ci_estimates_l1, axis=1)
    total_ci_estimates_u1 = np.concatenate(total_ci_estimates_u1, axis=1)

    # compute confidence intervals
    n = total_ci_estimates_l0.shape[1]
    u_vals0_ci = np.std(total_ci_estimates_u0, axis=1) / np.sqrt(n) * 2.241
    l_vals0_ci = np.std(total_ci_estimates_l0, axis=1) / np.sqrt(n) * 2.241

    u_vals1_ci = np.std(total_ci_estimates_u1, axis=1) / np.sqrt(n) * 2.241
    l_vals1_ci = np.std(total_ci_estimates_l1, axis=1) / np.sqrt(n) * 2.241

    avg_true_values_0 = np.array(avg_true_values_0)
    avg_true_values_1 = np.array(avg_true_values_1)

    avg_values_0 = np.mean(avg_values_0, axis=0)
    avg_values_1 = np.mean(avg_values_1, axis=0)

    avg_true_values_0 = np.mean(avg_true_values_0)
    avg_true_values_1 = np.mean(avg_true_values_1)

    u0 = avg_values_0[1]
    l0 = avg_values_0[0]

    u1 = avg_values_1[1]
    l1 = avg_values_1[0]

    # convert ci into np arrays
    u_vals0_ci = np.array(u_vals0_ci)
    l_vals0_ci = np.array(l_vals0_ci)
    
    u_vals1_ci = np.array(u_vals1_ci)
    l_vals1_ci = np.array(l_vals1_ci)

    # plot ground truth line
    plt.axhline(y=avg_true_values_0, color='r', label="True Rate: White", linestyle='dashed')
    plt.axhline(y=avg_true_values_1, color='b', label="True Rate: Black", linestyle='dashed')

    #plot our bound
    plt.plot(gammas, u0, label="White", color="red") # red line
    plt.plot(gammas, l0, color="red")

    # add confidence interval region as shaded
    plt.fill_between(gammas, u0 - u_vals0_ci, u0 + u_vals0_ci, color="red", alpha=0.2)
    plt.fill_between(gammas, l0 - l_vals0_ci, l0 + l_vals0_ci, color="red", alpha=0.2)

    # # plot bounds for group 1 in the same color - lower bound is dotted line, upper bound is solid line
    plt.plot(gammas, u1, label="Black", color="blue") # blue line
    plt.plot(gammas, l1, color="blue") # blue dotted line

    # add confidence interval region as shaded
    plt.fill_between(gammas, u1 - u_vals1_ci, u1 + u_vals1_ci, color="blue", alpha=0.2)
    plt.fill_between(gammas, l1 - l_vals1_ci, l1 + l_vals1_ci, color="blue", alpha=0.2)

    plt.xlabel("Gamma", fontsize=18)
    plt.ylabel("Treatment among needy", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=12.5)
    plt.savefig("semi_synth.jpg", format="jpg", transparent=True, bbox_inches="tight")
    # plt.show()
