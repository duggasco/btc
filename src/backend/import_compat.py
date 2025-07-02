"""
Import compatibility layer for local vs Docker environments
"""
import os
import sys

# Detect if we're in Docker (files are copied with different names)
IS_DOCKER = os.path.exists('backend_api.py') and not os.path.exists('src/backend/api/main.py')

if IS_DOCKER:
    # Docker environment - files are in the same directory with different names
    # No changes needed, imports work as-is
    pass
else:
    # Local environment - add src paths
    import os
    import sys
    
    # Get the project root (parent of src)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    # Add paths
    sys.path.insert(0, os.path.join(project_root, 'src'))
    sys.path.insert(0, os.path.join(project_root, 'src', 'backend'))
    
    # Create import aliases for compatibility
    try:
        import models.paper_trading as paper_trading_persistence
        import services.data_fetcher as external_data_fetcher
        import models.database as database_models
        import models.lstm as lstm_model
        import services.integration as integration
        import services.backtesting as backtesting_system
        import services.notifications as discord_notifications
        
        # Add to sys.modules for import compatibility
        sys.modules['paper_trading_persistence'] = paper_trading_persistence
        sys.modules['external_data_fetcher'] = external_data_fetcher
        sys.modules['database_models'] = database_models
        sys.modules['lstm_model'] = lstm_model
        sys.modules['integration'] = integration
        sys.modules['backtesting_system'] = backtesting_system
        sys.modules['discord_notifications'] = discord_notifications
        
    except ImportError as e:
        print(f"Warning: Could not set up import compatibility: {e}")
