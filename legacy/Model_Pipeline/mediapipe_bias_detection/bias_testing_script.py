"""
Bias Testing Script for MediaPipe with MLflow Tracking
"""

import pandas as pd
from pathlib import Path
from body_language_analyzer import BodyLanguageAnalyzer
import mlflow
import mlflow.sklearn
from datetime import datetime

def run_bias_test(csv_path='video_labels.csv', data_folder='data', experiment_name='mediapipe_bias_detection'):
    """
    Run bias test on videos with MLflow tracking
    """
    
    # Set up MLflow
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=f"bias_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Load labels
        print("Loading video labels...")
        df = pd.read_csv(csv_path)
        
        # Clean up
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        df['video_name'] = df['video_name'].str.strip()
        
        print(f"Found {len(df)} videos to analyze\n")
        
        # Log parameters
        mlflow.log_param("total_videos", len(df))
        mlflow.log_param("data_folder", data_folder)
        mlflow.log_param("frame_interval", 30)
        
        # Log demographics distribution
        skin_tone_dist = df['skin_tone'].value_counts().to_dict()
        for tone, count in skin_tone_dist.items():
            mlflow.log_param(f"videos_{tone}_skin", count)
        
        # Initialize analyzer
        analyzer = BodyLanguageAnalyzer()
        
        # Add columns for results
        df['confidence'] = 0.0
        df['posture_score'] = 0.0
        df['detection_rate'] = 0.0
        df['keypoints_detected'] = 0.0
        df['status'] = ''
        
        # Process each video
        data_path = Path(data_folder)
        
        for idx, row in df.iterrows():
            video_name = row['video_name']
            
            if not video_name.endswith('.mp4'):
                video_name = f"{video_name}.mp4"
            
            video_path = data_path / video_name
            
            print(f"Processing {video_name}...")
            
            if not video_path.exists():
                print(f"  ❌ Video not found: {video_path}")
                df.loc[idx, 'status'] = 'not_found'
                continue
            
            try:
                results = analyzer.analyze_video(str(video_path), frame_interval=30)
                
                if 'error' in results:
                    print(f"  ⚠️  Error: {results['error']}")
                    df.loc[idx, 'status'] = 'error'
                else:
                    df.loc[idx, 'confidence'] = results['overall_confidence']
                    df.loc[idx, 'posture_score'] = results['avg_posture_score']
                    df.loc[idx, 'detection_rate'] = results['detection_rate']
                    df.loc[idx, 'keypoints_detected'] = results['avg_keypoints_detected']
                    df.loc[idx, 'status'] = 'success'
                    
                    print(f"  ✅ Confidence: {results['overall_confidence']:.2f}, "
                          f"Posture: {results['avg_posture_score']:.2f}")
            
            except Exception as e:
                print(f"  ❌ Exception: {e}")
                df.loc[idx, 'status'] = 'exception'
        
        # Save results
        output_file = 'bias_test_results.csv'
        df.to_csv(output_file, index=False)
        print(f"\n✅ Results saved to {output_file}")
        
        # Log CSV as artifact
        mlflow.log_artifact(output_file)
        
        # Analyze results
        print("\n" + "="*60)
        print("BIAS ANALYSIS RESULTS")
        print("="*60)
        
        successful = df[df['status'] == 'success']
        
        if len(successful) == 0:
            print("❌ No successful analyses to compare")
            mlflow.log_metric("success_rate", 0.0)
            return
        
        success_rate = len(successful) / len(df)
        mlflow.log_metric("success_rate", success_rate)
        
        print(f"\nSuccessfully analyzed: {len(successful)}/{len(df)} videos\n")
        
        # Group by skin tone
        print("CONFIDENCE SCORES BY SKIN TONE:")
        print("-" * 40)
        
        skin_tone_groups = successful.groupby('skin_tone')['confidence'].agg(['mean', 'count', 'std'])
        print(skin_tone_groups)
        
        # Log metrics by skin tone
        for tone in skin_tone_groups.index:
            avg_conf = skin_tone_groups.loc[tone, 'mean']
            mlflow.log_metric(f"confidence_{tone}_skin", avg_conf)
            mlflow.log_metric(f"keypoints_{tone}_skin", 
                            successful[successful['skin_tone']==tone]['keypoints_detected'].mean())
            mlflow.log_metric(f"detection_rate_{tone}_skin",
                            successful[successful['skin_tone']==tone]['detection_rate'].mean())
        
        # Detect bias
        print("\nBIAS DETECTION:")
        print("-" * 40)
        
        if len(skin_tone_groups) > 1:
            max_conf = skin_tone_groups['mean'].max()
            min_conf = skin_tone_groups['mean'].min()
            
            difference_pct = (max_conf - min_conf) / max_conf * 100
            
            print(f"Highest avg confidence: {max_conf:.2f}")
            print(f"Lowest avg confidence: {min_conf:.2f}")
            print(f"Difference: {difference_pct:.1f}%")
            
            # Log bias metrics
            mlflow.log_metric("max_confidence", max_conf)
            mlflow.log_metric("min_confidence", min_conf)
            mlflow.log_metric("bias_difference_pct", difference_pct)
            
            if difference_pct > 15:
                bias_detected = "yes"
                print("\n⚠️  BIAS DETECTED: >15% difference in confidence scores")
            else:
                bias_detected = "no"
                print("\n✅ No significant bias detected (<15% difference)")
            
            mlflow.log_param("bias_detected", bias_detected)
            mlflow.set_tag("bias_severity", "high" if difference_pct > 25 else "moderate" if difference_pct > 15 else "low")
        
        # Group by gender
        print("\n\nCONFIDENCE SCORES BY GENDER:")
        print("-" * 40)
        gender_groups = successful.groupby('gender')['confidence'].agg(['mean', 'count'])
        print(gender_groups)
        
        for gender in gender_groups.index:
            mlflow.log_metric(f"confidence_{gender}", gender_groups.loc[gender, 'mean'])
        
        # Group by lighting
        print("\n\nCONFIDENCE SCORES BY LIGHTING:")
        print("-" * 40)
        lighting_groups = successful.groupby('lighting_quality')['confidence'].agg(['mean', 'count'])
        print(lighting_groups)
        
        for lighting in lighting_groups.index:
            mlflow.log_metric(f"confidence_{lighting}_lighting", lighting_groups.loc[lighting, 'mean'])
        
        # Overall stats
        print("\n\nOVERALL STATISTICS:")
        print("-" * 40)
        avg_confidence = successful['confidence'].mean()
        avg_posture = successful['posture_score'].mean()
        avg_detection = successful['detection_rate'].mean()
        avg_keypoints = successful['keypoints_detected'].mean()
        
        print(f"Average confidence: {avg_confidence:.2f}")
        print(f"Average posture score: {avg_posture:.2f}")
        print(f"Average detection rate: {avg_detection:.1%}")
        print(f"Average keypoints detected: {avg_keypoints:.1f}/33")
        
        # Log overall metrics
        mlflow.log_metric("avg_confidence_overall", avg_confidence)
        mlflow.log_metric("avg_posture_score", avg_posture)
        mlflow.log_metric("avg_detection_rate", avg_detection)
        mlflow.log_metric("avg_keypoints_detected", avg_keypoints)
        
        # Log tags
        mlflow.set_tag("model", "MediaPipe Pose v0.10.14")
        mlflow.set_tag("test_date", datetime.now().strftime('%Y-%m-%d'))
        
        return df


if __name__ == "__main__":
    # Install mlflow first: pip install mlflow
    
    results = run_bias_test(
        csv_path='video_labels.csv',
        data_folder='data',
        experiment_name='mediapipe_bias_detection'
    )
    
    print("\n" + "="*60)
    print("Testing complete! Check 'bias_test_results.csv' for full results.")
    print("View MLflow dashboard: mlflow ui")
    print("="*60)