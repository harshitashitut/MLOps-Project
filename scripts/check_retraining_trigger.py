"""
Retraining Trigger Checker
Determines if model retraining should be triggered based on various conditions.

Usage: python scripts/check_retraining_trigger.py
Returns: Exit code 0 if should retrain, 1 if not
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger


class RetrainingTrigger:
    def __init__(self):
        """Initialize the trigger checker."""
        self.triggers = {
            'feedback_threshold': False,
            'scheduled_time': False,
            'data_drift': False,
            'performance_degradation': False
        }
        
        # Configuration
        self.feedback_threshold = 100  # Number of user corrections needed
        self.days_since_last_training = 7  # Weekly schedule
        self.drift_threshold = 0.3  # 30% drift triggers retraining
        self.performance_drop_threshold = 0.05  # 5% accuracy drop
    
    def check_feedback_threshold(self) -> bool:
        """
        Check if enough user feedback has been collected.
        
        User feedback is when users correct the model's predictions.
        After 100 corrections, we should retrain to learn from these.
        """
        feedback_file = Path("../data/user_feedback/corrections.json")
        
        if not feedback_file.exists():
            logger.info(" No feedback file found")
            return False
        
        try:
            with open(feedback_file, 'r') as f:
                feedback_data = json.load(f)
            
            # Count new feedback since last training
            last_training_file = Path("model_registry/version.json")
            
            if last_training_file.exists():
                with open(last_training_file, 'r') as f:
                    version_data = json.load(f)
                
                if version_data.get('history'):
                    last_training_date = datetime.fromisoformat(
                        version_data['history'][-1]['date']
                    )
                    
                    # Count feedback after last training
                    new_feedback = sum(
                        1 for item in feedback_data
                        if datetime.fromisoformat(item['timestamp']) > last_training_date
                    )
                else:
                    new_feedback = len(feedback_data)
            else:
                new_feedback = len(feedback_data)
            
            logger.info(f" New feedback: {new_feedback}/{self.feedback_threshold}")
            
            return new_feedback >= self.feedback_threshold
            
        except Exception as e:
            logger.error(f" Error checking feedback: {e}")
            return False
    
    def check_scheduled_time(self) -> bool:
        """
        Check if scheduled retraining time has arrived.
        
        By default, retrain weekly (every 7 days).
        """
        last_training_file = Path("model_registry/version.json")
        
        if not last_training_file.exists():
            logger.info(" No previous training found - triggering first training")
            return True
        
        try:
            with open(last_training_file, 'r') as f:
                version_data = json.load(f)
            
            if not version_data.get('history'):
                logger.info(" No training history - triggering first training")
                return True
            
            last_training_date = datetime.fromisoformat(
                version_data['history'][-1]['date']
            )
            
            days_since = (datetime.now() - last_training_date).days
            
            logger.info(f" Days since last training: {days_since}/{self.days_since_last_training}")
            
            return days_since >= self.days_since_last_training
            
        except Exception as e:
            logger.error(f" Error checking schedule: {e}")
            return False
    
    def check_data_drift(self) -> bool:
        """
        Check if significant data drift has been detected.
        
        Data drift = when the input data distribution changes over time.
        Example: Users start submitting different types of slides.
        """
        drift_file = Path("monitoring/drift_reports/latest_drift.json")
        
        if not drift_file.exists():
            logger.info("📊 No drift report found")
            return False
        
        try:
            with open(drift_file, 'r') as f:
                drift_data = json.load(f)
            
            # Check if drift exceeds threshold
            drift_score = drift_data.get('drift_score', 0)
            
            logger.info(f"🔄 Data drift score: {drift_score:.2%} (threshold: {self.drift_threshold:.2%})")
            
            return drift_score > self.drift_threshold
            
        except Exception as e:
            logger.error(f" Error checking drift: {e}")
            return False
    
    def check_performance_degradation(self) -> bool:
        """
        Check if model performance has degraded significantly.
        
        Performance degradation = model accuracy dropping over time.
        This can happen as user needs change or edge cases appear.
        """
        metrics_file = Path("monitoring/performance_metrics/latest_metrics.json")
        
        if not metrics_file.exists():
            logger.info(" No performance metrics found")
            return False
        
        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            # Check if accuracy has dropped
            current_accuracy = metrics.get('current_accuracy', 1.0)
            baseline_accuracy = metrics.get('baseline_accuracy', 0.85)
            
            degradation = baseline_accuracy - current_accuracy
            
            logger.info(f"Performance degradation: {degradation:.2%} (threshold: {self.performance_drop_threshold:.2%})")
            
            return degradation > self.performance_drop_threshold
            
        except Exception as e:
            logger.error(f" Error checking performance: {e}")
            return False
    
    def check_all_triggers(self) -> dict:
        """Check all retraining triggers."""
        logger.info("="*60)
        logger.info(" CHECKING RETRAINING TRIGGERS")
        logger.info("="*60)
        
        self.triggers['feedback_threshold'] = self.check_feedback_threshold()
        self.triggers['scheduled_time'] = self.check_scheduled_time()
        self.triggers['data_drift'] = self.check_data_drift()
        self.triggers['performance_degradation'] = self.check_performance_degradation()
        
        return self.triggers
    
    def should_retrain(self) -> bool:
        """
        Determine if retraining should be triggered.
        
        Retrains if ANY trigger is activated.
        """
        triggers = self.check_all_triggers()
        
        # Retrain if any trigger is activated
        should_retrain = any(triggers.values())
        
        logger.info("\n" + "="*60)
        logger.info("TRIGGER SUMMARY")
        logger.info("="*60)
        
        for trigger, status in triggers.items():
            status_icon = "" if status else ""
            trigger_name = trigger.replace('_', ' ').title()
            logger.info(f"{status_icon} {trigger_name:30s}: {'TRIGGERED' if status else 'Not triggered'}")
        
        logger.info("="*60)
        
        if should_retrain:
            logger.info(" DECISION: RETRAIN MODEL")
            logger.info("   One or more triggers activated")
        else:
            logger.info(" DECISION: SKIP RETRAINING")
            logger.info("   No triggers activated")
        
        logger.info("="*60)
        
        return should_retrain


def main():
    """Main execution function."""
    checker = RetrainingTrigger()
    should_retrain = checker.should_retrain()
    
    # Write result for GitHub Actions
    result_file = Path("trigger_result.txt")
    with open(result_file, "w") as f:
        f.write("true" if should_retrain else "false")
    
    logger.info(f"\nResult written to: {result_file}")
    
    # Exit code: 0 if should retrain, 1 if not
    import sys
    sys.exit(0 if should_retrain else 1)


if __name__ == "__main__":
    main()
    