"""
Example: Complete Analysis Pipeline Usage
Demonstrates how to use all the new features in Fair-or-Foul
"""

from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from fairorfoul import CompleteAnalysisPipeline
from fairorfoul import StatisticalAnalyzer, BiasReportGenerator
from fairorfoul import MartialArtsValidator
from fairorfoul import VisionPipeline, KinematicAnalyzer


def example_video_analysis():
    """Example: Analyze video with computer vision and kinematics"""
    print("=" * 60)
    print("Example 1: Video Analysis with CV and Kinematics")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = CompleteAnalysisPipeline(
        sport="basketball",
        device="auto"
    )
    
    # Analyze video
    results = pipeline.analyze_video(
        video_path="data/raw/game_video.mp4",
        output_video_path="data/processed/annotated_video.mp4",
        output_data_path="data/processed/video_analysis.json"
    )
    
    print(f"Total frames processed: {results['vision_results']['total_frames']}")
    print(f"Players tracked: {results['vision_results']['total_tracks']}")
    print(f"Collisions detected: {results['vision_results']['total_collisions']}")
    print(f"Potential fouls: {results['summary']['potential_fouls']}")
    print(f"Whistle events: {results['summary']['whistle_events_count']}")


def example_statistical_analysis():
    """Example: Statistical analysis with Kruskal-Wallis, Mann-Whitney U, Dublin Delta"""
    print("\n" + "=" * 60)
    print("Example 2: Statistical Analysis")
    print("=" * 60)
    
    import pandas as pd
    
    # Load data
    df = pd.read_csv("data/raw/calls.csv")
    
    # Initialize analyzer
    analyzer = StatisticalAnalyzer(alpha=0.05)
    report_generator = BiasReportGenerator(analyzer)
    
    # Generate comprehensive report
    report = report_generator.generate_report(
        df,
        include_dublin_delta=True,
        include_league_comparison=False
    )
    
    print(f"Bias detected: {report['summary']['bias_detected']}")
    print(f"Referee variance: {report['summary']['referee_variance']}")
    print(f"Team bias: {report['summary']['team_bias']}")
    
    if report['dublin_delta']:
        print(f"Dublin Delta: {report['dublin_delta']['mean_delta']:.3f}")
        print(f"Dublin Delta bias: {report['dublin_delta']['bias_detected']}")


def example_league_comparison():
    """Example: Compare leagues (LOI vs La Liga)"""
    print("\n" + "=" * 60)
    print("Example 3: League Comparison (LOI vs La Liga)")
    print("=" * 60)
    
    import pandas as pd
    from fairorfoul import StatisticalAnalyzer
    
    # Load league data
    loi_data = pd.read_csv("data/raw/loi_calls.csv")
    laliga_data = pd.read_csv("data/raw/laliga_calls.csv")
    
    # Compare leagues
    analyzer = StatisticalAnalyzer()
    comparison = analyzer.league_comparison(loi_data, laliga_data, metric_column='call_rate')
    
    print(f"LOI mean call rate: {comparison['loi_stats']['mean']:.3f}")
    print(f"La Liga mean call rate: {comparison['laliga_stats']['mean']:.3f}")
    print(f"Significant difference: {comparison['significant_difference']}")
    print(f"Effect size (Cohen's d): {comparison['cohens_d']:.3f}")
    print(f"Effect size interpretation: {comparison['effect_size_interpretation']}")


def example_martial_arts_validation():
    """Example: Martial arts validation with UFC data"""
    print("\n" + "=" * 60)
    print("Example 4: Martial Arts Validation")
    print("=" * 60)
    
    validator = MartialArtsValidator()
    
    # Generate validation report
    report = validator.generate_validation_report(
        ufc_data_path="data/raw/ufc_data.csv",
        output_path="data/processed/ufc_validation.json"
    )
    
    print(f"Total fights analyzed: {report['data_summary']['total_fights']}")
    print(f"Unique referees: {report['data_summary']['unique_referees']}")
    print(f"Bias indicator: {report['referee_bias']['bias_indicator']}")
    print(f"Coefficient of variation: {report['referee_bias']['coefficient_of_variation']:.3f}")


def example_kinematic_analysis():
    """Example: Standalone kinematic analysis"""
    print("\n" + "=" * 60)
    print("Example 5: Kinematic Analysis")
    print("=" * 60)
    
    from fairorfoul import KinematicAnalyzer
    
    # Initialize analyzer
    analyzer = KinematicAnalyzer(fps=30.0, court_length_m=28.0, court_width_m=15.0)
    
    # Example trajectory (pixel positions)
    pixel_positions = [
        (100, 200), (105, 205), (110, 210), (115, 215),
        (120, 220), (125, 225), (130, 230)
    ]
    
    # Analyze trajectory
    analysis = analyzer.analyze_player_trajectory(pixel_positions)
    
    print(f"Max speed: {analysis['max_speed']:.2f} m/s")
    print(f"Max G-force: {analysis['max_gforce']:.2f} G")
    print(f"Total distance: {analysis['total_distance']:.2f} m")
    print(f"Whistle threshold: {analysis['whistle_threshold']['threshold']:.3f}")
    print(f"Should whistle: {analysis['whistle_threshold']['should_whistle']}")


def example_complete_pipeline():
    """Example: Complete end-to-end analysis"""
    print("\n" + "=" * 60)
    print("Example 6: Complete Pipeline")
    print("=" * 60)
    
    pipeline = CompleteAnalysisPipeline(sport="basketball")
    
    # Generate complete report
    report = pipeline.generate_complete_report(
        video_path="data/raw/game_video.mp4",
        csv_path="data/raw/calls.csv",
        ufc_data_path="data/raw/ufc_data.csv",
        output_dir="reports/complete_analysis"
    )
    
    print("Complete analysis report generated!")
    print(f"Video analysis: {len(report['video_analysis']) > 0}")
    print(f"Statistical analysis: {len(report['statistical_analysis']) > 0}")
    print(f"Martial arts validation: {len(report['martial_arts_validation']) > 0}")


if __name__ == "__main__":
    print("Fair-or-Foul Complete Analysis Examples")
    print("=" * 60)
    print("\nNote: These examples require data files to be present.")
    print("Modify paths as needed for your data.\n")
    
    # Uncomment the examples you want to run:
    # example_video_analysis()
    # example_statistical_analysis()
    # example_league_comparison()
    # example_martial_arts_validation()
    # example_kinematic_analysis()
    # example_complete_pipeline()
    
    print("\nExamples defined. Uncomment in __main__ to run.")

