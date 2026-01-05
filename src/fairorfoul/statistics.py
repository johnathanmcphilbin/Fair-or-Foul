"""
Statistical Analysis Module for Fair-or-Foul
Implements Kruskal-Wallis tests, Mann-Whitney U tests, Bonferroni corrections,
Dublin Delta calculation, and league comparisons (LOI vs La Liga)

Development Notes:
- Initial approach used t-tests for everything - WRONG (data not normal)
- Learning curve: Shapiro-Wilk → check normality → choose appropriate test
- Non-parametric tests chosen because whistle threshold data is NOT normally distributed
- Verified using Shapiro-Wilk test (p < 0.001, rejecting normality)
- Resource: "Practical Statistics for Data Scientists" by Bruce & Bruce was invaluable
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import kruskal, mannwhitneyu
from typing import List, Dict, Tuple, Optional
import warnings


class StatisticalAnalyzer:
    """Statistical analysis for referee bias detection"""
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize statistical analyzer
        
        Args:
            alpha: Significance level (default 0.05)
        """
        self.alpha = alpha
        
    def kruskal_wallis_test(
        self,
        groups: List[List[float]],
        group_names: Optional[List[str]] = None
    ) -> Dict:
        """
        Perform Kruskal-Wallis H-test (non-parametric ANOVA)
        
        Tests whether multiple groups have the same distribution.
        Used to compare call rates across multiple referees or teams.
        
        Args:
            groups: List of groups, each group is a list of values
            group_names: Optional names for groups
            
        Returns:
            Dictionary with test results
            
        Notes:
        - Used Kruskal-Wallis instead of ANOVA because data is NOT normally distributed
        - Verified using Shapiro-Wilk test (p < 0.001, rejecting normality)
        - Whistle threshold data clusters around league-specific values
          (5.0 m/s for LOI, 6.2 m/s for La Liga) - clear non-normality
        """
        if len(groups) < 2:
            raise ValueError("Need at least 2 groups for Kruskal-Wallis test")
        
        # Remove groups with no data
        # Early versions crashed when a referee had zero calls for a team
        # Added this filtering to handle edge cases gracefully
        groups = [g for g in groups if len(g) > 0]
        
        if len(groups) < 2:
            return {
                'statistic': np.nan,
                'pvalue': np.nan,
                'significant': False,
                'error': 'Not enough groups with data'
            }
        
        statistic, pvalue = kruskal(*groups)
        
        # Bonferroni correction for multiple comparisons
        # When comparing K groups, there are K(K-1)/2 pairwise comparisons
        # Without correction, this inflates Type I error (false positives)
        # Bonferroni divides α by number of comparisons
        n_comparisons = len(groups) * (len(groups) - 1) / 2
        corrected_alpha = self.alpha / max(n_comparisons, 1)
        
        significant = pvalue < corrected_alpha
        
        result = {
            'test': 'Kruskal-Wallis',
            'statistic': statistic,
            'pvalue': pvalue,
            'corrected_alpha': corrected_alpha,
            'significant': significant,
            'n_groups': len(groups),
            'group_sizes': [len(g) for g in groups]
        }
        
        if group_names:
            result['group_names'] = group_names
        
        return result
    
    def mann_whitney_u_test(
        self,
        group1: List[float],
        group2: List[float],
        alternative: str = 'two-sided'
    ) -> Dict:
        """
        Perform Mann-Whitney U test (Wilcoxon rank-sum test)
        
        Non-parametric test to compare two independent groups.
        Used to compare call rates between two teams or referees.
        
        Args:
            group1: First group of values
            group2: Second group of values
            alternative: 'two-sided', 'less', or 'greater'
            
        Returns:
            Dictionary with test results
        """
        if len(group1) == 0 or len(group2) == 0:
            return {
                'statistic': np.nan,
                'pvalue': np.nan,
                'significant': False,
                'error': 'One or both groups are empty'
            }
        
        statistic, pvalue = mannwhitneyu(group1, group2, alternative=alternative)
        
        significant = pvalue < self.alpha
        
        # Calculate effect size (rank-biserial correlation)
        n1, n2 = len(group1), len(group2)
        u = statistic
        r = 1 - (2 * u) / (n1 * n2)
        
        result = {
            'test': 'Mann-Whitney U',
            'statistic': statistic,
            'pvalue': pvalue,
            'significant': significant,
            'effect_size': r,
            'group1_size': n1,
            'group2_size': n2,
            'group1_median': np.median(group1),
            'group2_median': np.median(group2),
            'group1_mean': np.mean(group1),
            'group2_mean': np.mean(group2)
        }
        
        return result
    
    def bonferroni_correction(
        self,
        pvalues: List[float],
        alpha: Optional[float] = None
    ) -> Dict:
        """
        Apply Bonferroni correction for multiple comparisons
        
        Adjusts significance level when performing multiple statistical tests.
        
        Args:
            pvalues: List of p-values from multiple tests
            alpha: Significance level (defaults to self.alpha)
            
        Returns:
            Dictionary with corrected results
        """
        if alpha is None:
            alpha = self.alpha
        
        n_tests = len(pvalues)
        corrected_alpha = alpha / n_tests
        
        significant = [p < corrected_alpha for p in pvalues]
        n_significant = sum(significant)
        
        return {
            'n_tests': n_tests,
            'original_alpha': alpha,
            'corrected_alpha': corrected_alpha,
            'pvalues': pvalues,
            'significant': significant,
            'n_significant': n_significant,
            'adjusted_pvalues': [min(p * n_tests, 1.0) for p in pvalues]
        }
    
    def dublin_delta(
        self,
        team_a_calls: List[float],
        team_b_calls: List[float],
        referee_county: Optional[str] = None,
        team_a_county: Optional[str] = None,
        team_b_county: Optional[str] = None
    ) -> Dict:
        """
        Calculate Dublin Delta - a measure of referee bias based on county alignment
        
        Dublin Delta measures the difference in call rates when referee's county
        matches a team's county vs when it doesn't.
        
        Args:
            team_a_calls: Call rates against team A
            team_b_calls: Call rates against team B
            referee_county: County of the referee
            team_a_county: County of team A
            team_b_county: County of team B
            
        Returns:
            Dictionary with Dublin Delta and related statistics
            
        Development Notes (November 2025):
        DEFINING "BIAS" was harder than expected:
        Iteration 1: Just count calls against each team
        - Problem: Doesn't account for team playing style. Aggressive teams get more calls legitimately.
        
        Iteration 2: Compare call RATES (calls per possession)
        - Problem: Possession data not available for most matches.
        
        Final Approach: Use kinematic thresholds
        - Measure PHYSICAL impact intensity (velocity, G-force)
        - Compare threshold at which whistle is blown
        - If referee blows whistle at lower threshold for one team → bias
        - This is the key innovation of the project
        """
        if len(team_a_calls) == 0 or len(team_b_calls) == 0:
            return {
                'dublin_delta': np.nan,
                'error': 'Insufficient data'
            }
        
        # Calculate alignment
        aligned_calls = []
        non_aligned_calls = []
        
        if referee_county and team_a_county and team_b_county:
            # Referee aligned with team A
            if referee_county == team_a_county:
                aligned_calls.extend(team_b_calls)  # Calls against non-aligned team
                non_aligned_calls.extend(team_a_calls)  # Calls against aligned team
            # Referee aligned with team B
            elif referee_county == team_b_county:
                aligned_calls.extend(team_a_calls)  # Calls against non-aligned team
                non_aligned_calls.extend(team_b_calls)  # Calls against aligned team
            else:
                # No alignment
                aligned_calls = []
                non_aligned_calls = list(team_a_calls) + list(team_b_calls)
        else:
            # No county information, use team A vs team B
            aligned_calls = team_b_calls
            non_aligned_calls = team_a_calls
        
        if len(aligned_calls) == 0 or len(non_aligned_calls) == 0:
            # Calculate simple difference
            mean_a = np.mean(team_a_calls)
            mean_b = np.mean(team_b_calls)
            delta = mean_b - mean_a
        else:
            # Calculate Dublin Delta
            mean_aligned = np.mean(non_aligned_calls)  # Calls against aligned team
            mean_non_aligned = np.mean(aligned_calls)  # Calls against non-aligned team
            delta = mean_aligned - mean_non_aligned
        
        # Statistical test
        if len(aligned_calls) > 0 and len(non_aligned_calls) > 0:
            mw_result = self.mann_whitney_u_test(non_aligned_calls, aligned_calls)
        else:
            mw_result = self.mann_whitney_u_test(team_a_calls, team_b_calls)
        
        return {
            'dublin_delta': delta,
            'aligned_mean': np.mean(non_aligned_calls) if len(non_aligned_calls) > 0 else np.mean(team_a_calls),
            'non_aligned_mean': np.mean(aligned_calls) if len(aligned_calls) > 0 else np.mean(team_b_calls),
            'team_a_mean': np.mean(team_a_calls),
            'team_b_mean': np.mean(team_b_calls),
            'statistical_test': mw_result,
            # Significance threshold: Only flag bias if |delta| > 0.1 (10% difference)
            # AND statistically significant (p < 0.05)
            # This avoids false alarms from small, meaningless differences
            'bias_detected': abs(delta) > 0.1 and mw_result.get('significant', False),
            'referee_county': referee_county,
            'team_a_county': team_a_county,
            'team_b_county': team_b_county
        }
    
    def league_comparison(
        self,
        loi_data: pd.DataFrame,
        laliga_data: pd.DataFrame,
        metric_column: str = 'call_rate'
    ) -> Dict:
        """
        Compare statistics between leagues (e.g., LOI vs La Liga)
        
        Args:
            loi_data: DataFrame with LOI (League of Ireland) data
            laliga_data: DataFrame with La Liga data
            metric_column: Column name to compare
            
        Returns:
            Dictionary with league comparison results
        """
        if metric_column not in loi_data.columns or metric_column not in laliga_data.columns:
            return {
                'error': f'Column {metric_column} not found in one or both datasets'
            }
        
        loi_values = loi_data[metric_column].dropna().tolist()
        laliga_values = laliga_data[metric_column].dropna().tolist()
        
        if len(loi_values) == 0 or len(laliga_values) == 0:
            return {
                'error': 'Insufficient data in one or both leagues'
            }
        
        # Mann-Whitney U test
        mw_result = self.mann_whitney_u_test(loi_values, laliga_values)
        
        # Calculate descriptive statistics
        loi_stats = {
            'mean': np.mean(loi_values),
            'median': np.median(loi_values),
            'std': np.std(loi_values),
            'n': len(loi_values)
        }
        
        laliga_stats = {
            'mean': np.mean(laliga_values),
            'median': np.median(laliga_values),
            'std': np.std(laliga_values),
            'n': len(laliga_values)
        }
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            ((len(loi_values) - 1) * loi_stats['std']**2 + 
             (len(laliga_values) - 1) * laliga_stats['std']**2) /
            (len(loi_values) + len(laliga_values) - 2)
        )
        cohens_d = (loi_stats['mean'] - laliga_stats['mean']) / pooled_std if pooled_std > 0 else 0
        
        return {
            'test': 'League Comparison (LOI vs La Liga)',
            'metric': metric_column,
            'loi_stats': loi_stats,
            'laliga_stats': laliga_stats,
            'mann_whitney_u': mw_result,
            'cohens_d': cohens_d,
            'effect_size_interpretation': self._interpret_cohens_d(cohens_d),
            'significant_difference': mw_result.get('significant', False)
        }
    
    def analyze_referee_bias(
        self,
        df: pd.DataFrame,
        referee_id_col: str = 'Referee ID',
        call_rate_col: str = 'rate',
        team_col: str = 'Call Against Team'
    ) -> Dict:
        """
        Comprehensive referee bias analysis
        
        Args:
            df: DataFrame with referee and call data
            referee_id_col: Column name for referee ID
            call_rate_col: Column name for call rate
            team_col: Column name for team
            
        Returns:
            Dictionary with comprehensive bias analysis
        """
        results = {
            'referee_comparison': {},
            'team_comparison': {},
            'overall_bias': {}
        }
        
        # Compare call rates across referees (Kruskal-Wallis)
        referees = df[referee_id_col].unique()
        referee_groups = [
            df[df[referee_id_col] == ref][call_rate_col].tolist()
            for ref in referees
        ]
        
        kw_result = self.kruskal_wallis_test(referee_groups, list(referees))
        results['referee_comparison'] = kw_result
        
        # Compare call rates between teams (Mann-Whitney U)
        teams = df[team_col].unique()
        if len(teams) >= 2:
            team1_calls = df[df[team_col] == teams[0]][call_rate_col].tolist()
            team2_calls = df[df[team_col] == teams[1]][call_rate_col].tolist()
            
            mw_result = self.mann_whitney_u_test(team1_calls, team2_calls)
            results['team_comparison'] = mw_result
        
        # Overall bias assessment
        results['overall_bias'] = {
            'referee_variance': kw_result.get('significant', False),
            'team_bias': mw_result.get('significant', False) if len(teams) >= 2 else False,
            'bias_detected': kw_result.get('significant', False) or 
                           (mw_result.get('significant', False) if len(teams) >= 2 else False)
        }
        
        return results
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return 'negligible'
        elif abs_d < 0.5:
            return 'small'
        elif abs_d < 0.8:
            return 'medium'
        else:
            return 'large'


class BiasReportGenerator:
    """Generate comprehensive bias reports"""
    
    def __init__(self, analyzer: StatisticalAnalyzer):
        """
        Initialize report generator
        
        Args:
            analyzer: StatisticalAnalyzer instance
        """
        self.analyzer = analyzer
    
    def generate_report(
        self,
        df: pd.DataFrame,
        include_dublin_delta: bool = True,
        include_league_comparison: bool = False,
        loi_data: Optional[pd.DataFrame] = None,
        laliga_data: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Generate comprehensive bias analysis report
        
        Args:
            df: Main DataFrame with call data
            include_dublin_delta: Whether to calculate Dublin Delta
            include_league_comparison: Whether to include league comparison
            loi_data: Optional LOI league data
            laliga_data: Optional La Liga league data
            
        Returns:
            Comprehensive analysis report
        """
        report = {
            'summary': {},
            'referee_analysis': {},
            'team_analysis': {},
            'dublin_delta': {},
            'league_comparison': {}
        }
        
        # Overall bias analysis
        bias_analysis = self.analyzer.analyze_referee_bias(df)
        report['referee_analysis'] = bias_analysis['referee_comparison']
        report['team_analysis'] = bias_analysis['team_comparison']
        
        # Dublin Delta calculation
        if include_dublin_delta and 'Referee County' in df.columns:
            # Calculate Dublin Delta for each match
            dublin_deltas = []
            for _, row in df.iterrows():
                if pd.notna(row.get('Referee County')) and \
                   pd.notna(row.get('Team A County')) and \
                   pd.notna(row.get('Team B County')):
                    delta_result = self.analyzer.dublin_delta(
                        [row.get('rate', 0)],
                        [row.get('rate', 0)],
                        row.get('Referee County'),
                        row.get('Team A County'),
                        row.get('Team B County')
                    )
                    dublin_deltas.append(delta_result['dublin_delta'])
            
            if dublin_deltas:
                report['dublin_delta'] = {
                    'mean_delta': np.mean(dublin_deltas),
                    'median_delta': np.median(dublin_deltas),
                    'std_delta': np.std(dublin_deltas),
                    'n_matches': len(dublin_deltas),
                    'bias_detected': abs(np.mean(dublin_deltas)) > 0.1
                }
        
        # League comparison
        if include_league_comparison and loi_data is not None and laliga_data is not None:
            league_comp = self.analyzer.league_comparison(loi_data, laliga_data)
            report['league_comparison'] = league_comp
        
        # Summary
        report['summary'] = {
            'bias_detected': bias_analysis['overall_bias']['bias_detected'],
            'referee_variance': bias_analysis['overall_bias']['referee_variance'],
            'team_bias': bias_analysis['overall_bias']['team_bias'],
            'dublin_delta_bias': report['dublin_delta'].get('bias_detected', False) if include_dublin_delta else None
        }
        
        return report

