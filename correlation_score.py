import pandas as pd
import scipy.stats as stats

def calculate_spearman_correlation(X, Y):
    return stats.spearmanr(X, Y)[0]

def calculate_spearman_correlation_p(X, Y):
    return stats.spearmanr(X, Y)[1]
if __name__=='__main__':
    data = pd.read_excel(r'D:\data.xlsx', header=0)
    X_urea= data['urea']
    X_Cl = data['Cl']
    X_HCO = data['HCO']
    X_NH= data['NH']
    X_temp= data['temp']
    X_PDS=data['PDS']
    y = data['kobs']
    print('urea-y correlation score1',calculate_spearman_correlation(X_urea, y),'P' ,calculate_spearman_correlation_p(X_urea, y))
    print('Cl-y correlation score1', calculate_spearman_correlation(X_Cl, y), 'P',
          calculate_spearman_correlation_p(X_Cl, y))
    print('HCO-y correlation score1', calculate_spearman_correlation(X_HCO, y), 'P',
          calculate_spearman_correlation_p(X_HCO, y))
    print('NH-y correlation score1', calculate_spearman_correlation(X_NH, y), 'P',
          calculate_spearman_correlation_p(X_NH, y))
    print('temp-y correlation score1', calculate_spearman_correlation(X_temp, y), 'P',
          calculate_spearman_correlation_p(X_temp, y))
    print('PDS-y correlation score1', calculate_spearman_correlation(X_PDS, y), 'P',
          calculate_spearman_correlation_p(X_PDS, y))

pearson=data.corr()
print(pearson)
kendall=data.corr('kendall')
print(kendall)
spearman=data.corr('spearman')
print(spearman)