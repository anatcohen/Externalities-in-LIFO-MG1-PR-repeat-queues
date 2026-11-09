import numpy as np
import pandas as pd
from itertools import combinations_with_replacement

A = 'A'
B = 'B'

class Externality():
    def __init__(self, r : float, lam : float, service_times : np.array):
        # Initialise parameters
        self.r = r
        self.lam = lam
        self.beta = 1/(31-2*self.lam)
        self.service_times = service_times
        self.k = self.service_times.shape[0] - 1

        self._test_input()

        # Generate sample space
        H = self._gen_Hs()
        self.H_df = self._convert_H_to_df(H=H)

        # Add relevant terms for externality calculation
        self._add_bounds()

    def _test_input(self):
        if self.k <= 0:
            raise ValueError('Number of customers in system should be larger than 0.')
        if self.r <= 0:
            raise ValueError('r should be positive.')
        if self.lam <= 0:
            raise ValueError('lam is expected to be strictly positive.')
        if (self.service_times <= 0).any():
            raise ValueError('Service times are expected to be strictly positive.')
        if self.service_times[1] <= self.r:
            raise ValueError('The remaining service time for c1 must be smaller than its overall service time.')

    def _gen_Hs(self) -> np.array:
        """
        Creates all possible orderings of H in system A (with c0) and B (without c0).

        :return: A numpy matrix with 2k+1 columns where each row is a possible ordering. The first n+1 columns belong to A and the following n columns belong to B.
        """
        A = list(combinations_with_replacement(range(2 * self.k + 2), self.k + 1))
        B = list(combinations_with_replacement(range(2 * self.k + 2), self.k))
        matrix = [
            list(a) + list(b)
            for a in A
            for b in B
            if all(a[i+1] >= b[i] for i in range(self.k))
            and set(range(1, max(a + b) + 1)).issubset(a + b)
        ]
        return np.array(matrix)

    def _convert_H_to_df(self, H : np.array) -> pd.DataFrame:
        """
        Converts matrix H to a long DataFrame. Rows are identified by index columns 'config', 'H_index', 'system.

        :param H: H matrix generated in gen_Hs.
        """
        n_configs, total_customers = H.shape

        df = pd.DataFrame({
            "config": np.repeat(np.arange(n_configs), total_customers),
            "H_index": H.ravel(),
            "customer": np.tile(np.concatenate([np.arange(self.k+1), np.arange(1, self.k+1)]), n_configs),
            "system": np.tile(np.concatenate([np.repeat([A], self.k + 1), np.repeat([B], self.k)]), n_configs),
            "service": np.tile(np.concatenate([self.service_times, self.service_times[1:]]), n_configs)
        })

        df.loc[(df["H_index"] == 0) & (df['customer'] == 1) & (df["system"] == B), 'service'] = self.r
        return df

    def _add_bounds(self):
        # Create helper columns
        self.H_df['cumul_total_service'] = self.H_df.groupby(['config', 'H_index', 'system'])['service'].cumsum()
        self.H_df = self.H_df.merge(self.H_df.loc[self.H_df['system'] == A, ['config', 'customer', 'cumul_total_service']], how='left', on=['config', 'customer'], suffixes=('', '_y')).rename(columns={'cumul_total_service_y': f'S_{A}'})
        self.H_df = self.H_df.merge(self.H_df.loc[self.H_df['system'] == B, ['config', 'customer', 'cumul_total_service']], how='left', on=['config', 'customer'], suffixes=('', '_y')).rename(columns={'cumul_total_service_y': f'S_{B}'})

        self.H_df = self.H_df.merge(self.H_df.loc[self.H_df['system'] == A, ['config', 'H_index', 'customer']], how='left', on=['config', 'customer'], suffixes=('', '_y')).rename(columns={'H_index_y': 'a'})
        self.H_df = self.H_df.merge(self.H_df.loc[self.H_df['system'] == B, ['config', 'H_index', 'customer']], how='left', on=['config', 'customer'], suffixes=('', '_y')).rename(columns={'H_index_y': 'b'})
        self.H_df['b'].bfill(inplace=True)


        # Check if the same customers are present in both systems for each (config, H_index)
        customers = self.H_df[self.H_df['customer'] != 0].groupby(['config', 'H_index' ,'system'])['customer'].agg(set).unstack('system')
        customers['same_customers'] = customers['A'] == customers['B']
        self.H_df ['same_customers'] = self.H_df.set_index(['config', 'H_index']).index.map(customers['same_customers'])

        # Add bounds
        bounds = self._calc_bounds()
        self.H_df = self.H_df.merge(bounds, on=['config', 'H_index'], how='left')

        # Filter to valid zeta<eta cases
        self.H_df = self.H_df[~(self.H_df['config'].isin(self.H_df.loc[self.H_df['eta'] <= self.H_df['zeta'], 'config']))]

        # Calculate terms for externality
        self.H_df['var_phi'] = self.H_df[f'S_{A}'] - self.H_df[f'S_{B}']
        self.H_df['var_rho'] = np.where(self.H_df['system'] == B, self.H_df['zeta'] - self.H_df[f'S_{B}'], 0)
        self.H_df['rho'] = self.H_df[f'S_{A}']

        self.H_df['has_H0'] = self.H_df.groupby('config')['H_index'].transform(lambda x: (x==0).any())
        # self.H_df['eta0'] = self.H_df.groupby('config').transform(lambda x: x.loc[x['H_index'] == 1, 'alpha'].unique[0])
        eta0_map = (self.H_df[self.H_df['H_index'] == 1].drop_duplicates('config').set_index('config')['alpha'])

        self.H_df['eta0'] = self.H_df['config'].map(eta0_map)
        self.H_df.loc[self.H_df['alpha_customer'] == 1, 'eta0'] = self.r
        # self.H_df['Y_sum'] = self._calc_exp_Y()
        self._calc_exp_Y()

        # Calculate probability
        self.H_df['prob_term'] = np.exp(-self.lam * (self.H_df['zeta'] - self.H_df['alpha'])) - \
                                 np.exp(-self.lam * (self.H_df['eta'] - self.H_df['alpha']))
        self.H_df.loc[self.H_df['H_index'] == 0, 'prob_term'] = np.exp(-self.lam * self.H_df['zeta']) - \
                                                                np.exp(-self.lam * self.H_df['eta'])
        self.H_df['prob'] = self.H_df['config'].map(self.H_df.drop_duplicates(['config', 'H_index']).groupby('config')['prob_term'].prod())
        self.H_df.loc[~self.H_df['has_H0'], 'prob'] *= (1- np.exp(-self.lam * self.H_df['eta0']))

        # Calculate externality
        # self.H_df['exter'] = self.H_df['var_rho'] + self.H_df['rho'] + self.H_df['mu'] + self.beta + self.H_df['Y_sum']
        self.H_df['exter'] = self.H_df['var_rho'] - self.H_df['zeta'] + self.H_df['rho'] + (self.H_df['a'] - self.H_df['b'])*self.beta + self.H_df['Y_sum']
        self.H_df.loc[self.H_df['a'] == self.H_df['b'], 'exter'] = self.H_df['var_phi']

        # Calculate expected externality
        self.H_df['expected_exter'] = self.H_df['prob'] * self.H_df['exter']

        # Remove irrelevant columns
        self.H_df.drop(columns=[f'S_{A}', f'S_{B}', 'Y_sum', 'has_H0', 'eta0'], inplace=True)

    def _calc_Y_m(self):
        # Calculate mu
        d = self.H_df['eta'] - self.H_df['zeta']
        finite = self.H_df['eta'] < np.inf

        mu = self.H_df['zeta'] + 1 / self.lam
        mu[finite] -= d[finite] / np.expm1(self.lam * d[finite])

        # Calculate alpha_{m+1}
        alpha_lookup = (
            self.H_df[['config', 'H_index', 'alpha']]
            .drop_duplicates(['config', 'H_index'])
            .set_index(['config', 'H_index'])['alpha']
        )

        next_class = pd.MultiIndex.from_arrays([
            self.H_df['config'],
            self.H_df['H_index'] + 1
        ])

        alpha_m = alpha_lookup.reindex(next_class).to_numpy()

        # Calculate E[Y_m]
        E_Qm = self.beta + 1 / self.lam
        E_Lm = np.exp(self.lam * alpha_m) - 1
        E_Vm = alpha_m / np.expm1(self.lam * alpha_m)
        # E_Vm = (alpha_m * np.exp(-self.lam * alpha_m)) / (1 - np.exp(-self.lam * alpha_m))
        E_Y = mu + (E_Lm * (E_Qm - E_Vm))

        self.H_df['Y_m'] = E_Y.fillna(0)
        self.H_df['Y_m'] = self.H_df['Y_m'].fillna(0)

    def _calc_exp_Y(self) -> pd.Series:
        self._calc_Y_m()

        classes = (
            self.H_df[['config', 'H_index', 'Y_m']]
            .drop_duplicates(['config', 'H_index'])
            .sort_values(['config', 'H_index'])
            .copy()
        )

        # The final class q has no Y_q; give it zero for the prefix calculation.
        classes['Y_m'] = classes['Y_m'].fillna(0.0)
        classes['prefix'] = (
                classes.groupby('config')['Y_m'].cumsum()
                - classes['Y_m']
        )

        prefix = classes.set_index(['config', 'H_index'])['prefix']

        a_key = pd.MultiIndex.from_arrays([
            self.H_df['config'], self.H_df['a'].astype(int)
        ])
        b_key = pd.MultiIndex.from_arrays([
            self.H_df['config'], self.H_df['b'].astype(int)
        ])

        self.H_df['Y_sum'] = (
                prefix.reindex(a_key).to_numpy()
                - prefix.reindex(b_key).to_numpy()
        )

        self.H_df['Y_sum'] = self.H_df['Y_sum'].fillna(0)

    def _calc_bounds(self) -> pd.DataFrame:
        """
        Calculated bounds required for externality calculations (zeta and eta).

        :return: TBD
        """
        tmp = (self.H_df.groupby(['config', 'system', 'H_index'])
               .agg(sum=('service', 'sum'),
                      s=('service', 'first'),
                      s_customer=('customer', 'first')))

        # Ensure both systems exist for every (config, H_index)
        idx = tmp.index.to_frame(index=False)
        combinations = idx[['config', 'H_index']].drop_duplicates()

        complete_idx = pd.MultiIndex.from_frame(
            combinations.merge(
                pd.DataFrame({'system': ['A', 'B']}),
                how='cross'
            )[['config', 'system', 'H_index']]
        )

        tmp = tmp.reindex(complete_idx).sort_index().reset_index()
        tmp['is_original'] = (tmp['sum'].notna()) & (tmp['s'].notna())
        tmp['sum'].fillna(0, inplace=True)
        tmp['next_s'] = tmp.groupby(['config', 'system'])['s'].transform(lambda x: x.shift(-1).bfill()).fillna(np.inf)
        tmp['extended_sum'] = tmp['sum'] + tmp['next_s']

        idx = tmp.groupby(['config', 'H_index'])['s'].idxmin()

        alpha_customers = (
            tmp.loc[idx, ['config', 'H_index', 's_customer']]
            .rename(columns={'s_customer': 'alpha_customer'})
        )

        bounds = (
            tmp.groupby(['config', 'H_index'])
            .agg(
                zeta=('sum', 'max'),
                eta=('extended_sum', 'min'),
                alpha=('s', 'min'),
            )
            .reset_index()
            .merge(
                alpha_customers,
                on=['config', 'H_index'],
                how='left'
            )
        )

        return bounds

    def calculate_total_expected_externality(self) -> float:
        return self.H_df.loc[self.H_df['system'] == B, 'expected_exter'].sum()

    def calculate_expected_externality(self):
        return self.H_df[self.H_df['system'] == B].groupby('customer')['expected_exter'].sum()

    def calc_zero_externality_probability(self):
        self.H_df['adj_prob'] = self.H_df['same_customers'] * (self.H_df['a'] == self.H_df['b']) * self.H_df['prob']
        return self.H_df[self.H_df['system']== B].groupby('customer')['adj_prob'].sum()

    def calc_first_zero_externality_probability(self):
        zero_exter_prob = self.calc_zero_externality_probability()
        return zero_exter_prob.diff().fillna(zero_exter_prob.iloc[0])


def get_customer_expected_externality(lams : list, service_times: np.array, r_per:list, beta: float=1.5):
    res = pd.DataFrame()

    for i, s in enumerate(service_times):
        print(s)
        t = np.array(s)/np.sum(s)
        for lam in lams:
            for rp in r_per:
                r = rp * t[1]
                exter = Externality(service_times=t, r=r, lam=lam)

                df = exter.H_df.copy()
                m =  df.loc[df['system'] == B, 'expected_exter'].min()
                if m< 0: print(m)

                prob_df = exter.calc_zero_externality_probability().rename('Zero Externality Probability').reset_index()
                first_prob = exter.calc_first_zero_externality_probability().rename('First Zero Externality Probability').reset_index()
                exter_df = exter.calculate_expected_externality().rename('Expected Externality').reset_index()
                # exter_df = exter_df.rename('Expected Externality').reset_index()

                # Add parameters to the DataFrame
                exter_df['lambda'] = lam
                exter_df['beta'] = beta
                exter_df['r'] = rp
                exter_df['Service Time'] = str(s)
                exter_df = exter_df.merge(prob_df, on=['customer'])
                exter_df = exter_df.merge(first_prob, on=['customer'])

                res = pd.concat([res, exter_df])

    return res


def get_expected_total_externality(lams : list, service_times: np.array, r_per: list):

    res = []
    for i, s in enumerate(service_times):
        print(s)
        t = np.array(s)/np.sum(s)
        for lam in lams:
            for rp in r_per:
                r = rp * t[1]
                exter = Externality(service_times=t, r=r, lam=lam).calculate_total_expected_externality()
                res.append({
                    'lambda': lam,
                    'Service Time': str(s),
                    'Expected Total Externality': exter,
                    'r': rp,
                })

    res = pd.DataFrame(res)

    return res
