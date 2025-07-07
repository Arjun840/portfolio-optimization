import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  TrendingUp, 
  Settings, 
  BarChart3, 
  PieChart, 
  Target, 
  Shield,
  LogOut,
  RefreshCw,
  Download,
  Info,
  Bookmark
} from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import authService from '../services/authService';
import toast from 'react-hot-toast';

// Import visualization components (we'll create these next)
import PortfolioChart from '../components/PortfolioChart';
import AllocationChart from '../components/AllocationChart';
import EfficientFrontierChart from '../components/EfficientFrontierChart';
import StockClusterView from '../components/StockClusterView';
import OptimizationResults from '../components/OptimizationResults';
import OptimizationControls from '../components/OptimizationControls';
import CustomPortfolioBuilder from '../components/CustomPortfolioBuilder';
import SavedPortfolios from '../components/SavedPortfolios';
import LoadingSpinner from '../components/LoadingSpinner';

const Dashboard = () => {
  const { user, logout } = useAuth();
  const [activeTab, setActiveTab] = useState('optimize');
  const [optimizationResult, setOptimizationResult] = useState(null);
  const [stocks, setStocks] = useState([]);
  const [clusters, setClusters] = useState([]);
  const [efficientFrontier, setEfficientFrontier] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [loadingStates, setLoadingStates] = useState({
    stocks: true,
    clusters: true,
    optimization: false,
    efficientFrontier: false,
  });

  // Load initial data
  useEffect(() => {
    loadInitialData();
  }, []);

  const loadInitialData = async () => {
    try {
      // Load stocks and clusters in parallel
      const [stocksResult, clustersResult] = await Promise.all([
        authService.getStocks(),
        authService.getClusters(),
      ]);

      if (stocksResult.success) {
        setStocks(stocksResult.data);
      } else {
        toast.error('Failed to load stocks data');
      }

      if (clustersResult.success) {
        setClusters(clustersResult.data);
      } else {
        toast.error('Failed to load clusters data');
      }
    } catch (error) {
      toast.error('Failed to load initial data');
    } finally {
      setLoadingStates(prev => ({
        ...prev,
        stocks: false,
        clusters: false,
      }));
    }
  };

  const handleOptimization = async (optimizationParams) => {
    setLoadingStates(prev => ({ ...prev, optimization: true }));
    
    try {
      const result = await authService.optimizePortfolio(optimizationParams);
      
      if (result.success) {
        setOptimizationResult(result.data);
        toast.success('Portfolio optimized successfully!');
        
        // Switch to results tab
        setActiveTab('results');
      } else {
        toast.error(result.error || 'Optimization failed');
      }
    } catch (error) {
      toast.error('Optimization failed. Please try again.');
    } finally {
      setLoadingStates(prev => ({ ...prev, optimization: false }));
    }
  };

  const loadEfficientFrontier = async (constraintLevel = 'enhanced', returnMethod = 'ml_enhanced') => {
    setLoadingStates(prev => ({ ...prev, efficientFrontier: true }));
    
    try {
      const result = await authService.getEfficientFrontier(constraintLevel, returnMethod);
      
      if (result.success) {
        setEfficientFrontier(result.data);
        setActiveTab('analysis');
      } else {
        toast.error('Failed to load efficient frontier');
      }
    } catch (error) {
      toast.error('Failed to load efficient frontier');
    } finally {
      setLoadingStates(prev => ({ ...prev, efficientFrontier: false }));
    }
  };

  const handleExportResults = () => {
    if (!optimizationResult) return;
    
    const dataStr = JSON.stringify(optimizationResult, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `portfolio-optimization-${optimizationResult.portfolio_id}.json`;
    link.click();
    URL.revokeObjectURL(url);
    toast.success('Results exported successfully!');
  };

  const tabs = [
    { id: 'optimize', label: 'Optimize', icon: Target },
    { id: 'custom', label: 'Custom Portfolio', icon: Settings },
    { id: 'saved', label: 'Saved Portfolios', icon: Bookmark },
    { id: 'results', label: 'Results', icon: BarChart3 },
    { id: 'analysis', label: 'Analysis', icon: PieChart },
    { id: 'stocks', label: 'Stocks', icon: TrendingUp },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <div className="h-8 w-8 bg-gradient-to-r from-blue-500 to-blue-600 rounded-lg flex items-center justify-center mr-3">
                <div className="relative">
                  <TrendingUp className="h-5 w-5 text-white" />
                  <div className="absolute -top-1 -right-1 h-3 w-3 bg-green-400 rounded-full animate-pulse"></div>
                </div>
              </div>
              <h1 className="text-xl font-semibold bg-gradient-to-r from-gray-900 to-blue-700 bg-clip-text text-transparent">
                PortfolioMax
              </h1>
            </div>
            
            <div className="flex items-center space-x-4">
              <span className="text-sm text-gray-600">
                Welcome, {user?.email || 'User'}
              </span>
              <button
                onClick={() => loadInitialData()}
                className="p-2 text-gray-400 hover:text-gray-600 transition-colors"
                title="Refresh Data"
              >
                <RefreshCw className="h-5 w-5" />
              </button>
              <button
                onClick={logout}
                className="flex items-center text-sm text-gray-600 hover:text-gray-900 transition-colors"
              >
                <LogOut className="h-4 w-4 mr-1" />
                Logout
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Tab Navigation */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 mb-8">
          <div className="border-b border-gray-200">
            <nav className="flex space-x-8 px-6" aria-label="Tabs">
              {tabs.map((tab) => {
                const Icon = tab.icon;
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`
                      flex items-center py-4 px-1 border-b-2 font-medium text-sm transition-colors
                      ${activeTab === tab.id
                        ? 'border-blue-500 text-blue-600'
                        : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                      }
                    `}
                  >
                    <Icon className="h-4 w-4 mr-2" />
                    {tab.label}
                  </button>
                );
              })}
            </nav>
          </div>

          {/* Tab Content */}
          <div className="p-6">
            {activeTab === 'optimize' && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
              >
                <div className="mb-6">
                  <h2 className="text-2xl font-bold text-gray-900 mb-2">
                    Portfolio Optimization
                  </h2>
                  <p className="text-gray-600">
                    Configure your optimization parameters to find the optimal portfolio allocation
                  </p>
                </div>
                
                <OptimizationControls
                  onOptimize={handleOptimization}
                  onLoadEfficientFrontier={loadEfficientFrontier}
                  isLoading={loadingStates.optimization}
                  stocks={stocks}
                  clusters={clusters}
                />
              </motion.div>
            )}

            {activeTab === 'custom' && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
              >
                <CustomPortfolioBuilder stocks={stocks} />
              </motion.div>
            )}

            {activeTab === 'saved' && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
              >
                <SavedPortfolios />
              </motion.div>
            )}

            {activeTab === 'results' && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
              >
                <div className="flex justify-between items-center mb-6">
                  <div>
                    <h2 className="text-2xl font-bold text-gray-900 mb-2">
                      Optimization Results
                    </h2>
                    <p className="text-gray-600">
                      Detailed analysis of your optimized portfolio
                    </p>
                  </div>
                  
                  {optimizationResult && (
                    <button
                      onClick={handleExportResults}
                      className="btn-secondary flex items-center"
                    >
                      <Download className="h-4 w-4 mr-2" />
                      Export Results
                    </button>
                  )}
                </div>

                {optimizationResult ? (
                  <div className="space-y-8">
                    <OptimizationResults result={optimizationResult} />
                    
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                      <AllocationChart data={optimizationResult} />
                      <PortfolioChart data={optimizationResult} />
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-12">
                    <Target className="h-16 w-16 text-gray-300 mx-auto mb-4" />
                    <h3 className="text-lg font-medium text-gray-900 mb-2">
                      No Optimization Results
                    </h3>
                    <p className="text-gray-500 mb-4">
                      Run an optimization to see your results here
                    </p>
                    <button
                      onClick={() => setActiveTab('optimize')}
                      className="btn-primary"
                    >
                      Start Optimization
                    </button>
                  </div>
                )}
              </motion.div>
            )}

            {activeTab === 'analysis' && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
              >
                <div className="mb-6">
                  <h2 className="text-2xl font-bold text-gray-900 mb-2">
                    Portfolio Analysis
                  </h2>
                  <p className="text-gray-600">
                    Advanced analytics and efficient frontier visualization
                  </p>
                </div>

                {efficientFrontier ? (
                  <div className="space-y-8">
                    <EfficientFrontierChart data={efficientFrontier} />
                    {optimizationResult && (
                      <div className="bg-blue-50 rounded-lg p-6">
                        <h3 className="text-lg font-semibold text-blue-900 mb-2">
                          Your Portfolio on the Efficient Frontier
                        </h3>
                        <p className="text-blue-700">
                          Your optimized portfolio achieves a Sharpe ratio of{' '}
                          <strong>{optimizationResult.metrics.sharpe_ratio.toFixed(3)}</strong>{' '}
                          with {(optimizationResult.metrics.expected_return * 100).toFixed(1)}% expected return
                          and {(optimizationResult.metrics.volatility * 100).toFixed(1)}% volatility.
                        </p>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="text-center py-12">
                    <PieChart className="h-16 w-16 text-gray-300 mx-auto mb-4" />
                    <h3 className="text-lg font-medium text-gray-900 mb-2">
                      No Analysis Data
                    </h3>
                    <p className="text-gray-500 mb-4">
                      Load the efficient frontier to see advanced analytics
                    </p>
                    <button
                      onClick={() => loadEfficientFrontier()}
                      className="btn-primary"
                      disabled={loadingStates.efficientFrontier}
                    >
                      {loadingStates.efficientFrontier ? (
                        <>
                          <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                          Loading...
                        </>
                      ) : (
                        'Load Efficient Frontier'
                      )}
                    </button>
                  </div>
                )}
              </motion.div>
            )}

            {activeTab === 'stocks' && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
              >
                <div className="mb-6">
                  <h2 className="text-2xl font-bold text-gray-900 mb-2">
                    Available Stocks & Clusters
                  </h2>
                  <p className="text-gray-600">
                    Explore the available assets and their cluster classifications
                  </p>
                </div>

                {loadingStates.stocks || loadingStates.clusters ? (
                  <LoadingSpinner />
                ) : (
                  <StockClusterView stocks={stocks} clusters={clusters} />
                )}
              </motion.div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard; 