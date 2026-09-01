import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd

def analyze_poisson(file, row_name: str):
    """Input .csv file and name of column with counts per second info. 
    Output the distribution of the data, mean, variance, dof, chi squared, p-value, and whether the null hypothesis that the data follows Poisson distribution is rejected at confidence level 0.05."""
    # Read csv file and find the row with counts per time interval.
    df = pd.read_csv(file)
    raw_counts = df[row_name]

    N_total = len(raw_counts)
    mu = np.mean(raw_counts)
    sample_var = np.var(raw_counts, ddof=1)
    
    print(f"Total time intervals collected: {N_total}")
    print(f"Estimated Mean: {mu:.4f}")
    print(f"Sample Variance: {sample_var:.4f}")


    min_counts = int(np.min(raw_counts))
    max_counts = int(np.max(raw_counts))


    #Create bins with 1 extra value to return observed array with right number of values
    bins = np.arange(min_counts, max_counts+2, 1)

    observed, bins = np.histogram(raw_counts, bins=bins)

    #Remove extraneous added bin value
    bins = bins[:-1]
   
    k = len(observed)


    #Calculates probability mass for each discrete bin integer range
    expected_prob = []
    for i in range(k):
        counts_in_bin = np.arange(bins[i], bins[i] + 1)
        prob = np.sum(stats.poisson.pmf(counts_in_bin, mu))
        expected_prob.append(prob)

    expected_prob = np.array(expected_prob)

    # Normalize probabilities
    expected_prob /= np.sum(expected_prob)
    expected = expected_prob * N_total

    #Computes chi squared
    chi2_stat = np.sum(((observed - expected)**2)/expected)
    
    #Degrees of freedom with one estimated parameter is number of observations minus 2.
    dof = k - 2
    
    # p-value = 1 - CDF(chi2, dof)
    p_value = 1.0 - stats.chi2.cdf(chi2_stat, dof)

    print(f"Degrees of Freedom: {dof}")
    print(f"Chi-Squared: {chi2_stat:.4f}")
    print(f"Reduced Chi-Squared: {chi2_stat / dof:.4f}")
    print(f"p-value: {p_value:.6e}")
   

    alpha = 0.05
    if p_value < alpha:
        print(f"Reject H0 at alpha = {alpha}.")
    else:
        print(f"Fail to reject H0 at significance level = {alpha}.")

    #Plots distribution of counts, displays chi squared, degrees of freedom, and p-value.
    plt.figure(figsize=(9, 5))
    plt.bar(bins, observed, width=1, edgecolor="black")
    plt.title(f'Distribution ($\chi^2 = {chi2_stat:.2f}$, $df = {dof}$, $p = {p_value:.3f}$)', fontsize=22)
    plt.xlabel('Counts per Second (1/s)', fontsize=20)
    plt.ylabel('Frequency', fontsize=20)
    plt.show()


analyze_poisson(r"C:\Users\Noah Graham\Noah Personal Workspace\PHYS 281L HW\noah nes cesium lab 7 excel.csv", "radiation")