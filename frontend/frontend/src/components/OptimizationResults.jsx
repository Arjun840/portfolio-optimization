import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  TrendingUp, 
  TrendingDown, 
  Shield, 
  Target, 
  DollarSign,
  BarChart3,
  AlertTriangle,
  CheckCircle,
  Clock,
  Bookmark,
  X
} from 'lucide-react';
import authService from '../services/authService';

const OptimizationResults = ({ result }) => {
  const [showSaveModal, setShowSaveModal] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);
  const [portfolioName, setPortfolioName] = useState('');
  const [portfolioDescription, setPortfolioDescription] = useState('');
  const [saveError, setSaveError] = useState('');

  if (!result) return null;

  const { 
    portfolio_id, 
    strategy, 
    total_amount, 
    holdings, 
    metrics, 
    optimization_details,
    created_at 
  } = result;

  const formatPercentage = (value) => `${(value * 100).toFixed(2)}%`;
  const formatCurrency = (value) => `$${value.toLocaleString()}`;
  const formatNumber = (value, decimals = 2) => value.toFixed(decimals);

  const handleSavePortfolio = async () => {
    if (!portfolioName.trim()) {
      setSaveError('Portfolio name is required');
      return;
    }

    try {
      setSaving(true);
      setSaveError('');
      
      const saveResult = await authService.savePortfolio(
        portfolioName.trim(),
        portfolioDescription.trim() || null,
        result
      );

      if (saveResult.success) {
        setSaveSuccess(true);
        setTimeout(() => {
          setShowSaveModal(false);
          setSaveSuccess(false);
          setPortfolioName('');
          setPortfolioDescription('');
        }, 2000);
      } else {
        setSaveError(saveResult.error || 'Failed to save portfolio');
      }
    } catch (error) {
      setSaveError('Failed to save portfolio');
    } finally {
      setSaving(false);
    }
  };

  const openSaveModal = () => {
    setPortfolioName(strategy || 'My Portfolio');
    setPortfolioDescription('');
    setSaveError('');
    setSaveSuccess(false);
    setShowSaveModal(true);
  };

  const getMetricColor = (value, thresholds) => {
    if (value >= thresholds.good) return 'text-green-600';
    if (value >= thresholds.okay) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getMetricIcon = (value, thresholds) => {
    if (value >= thresholds.good) return CheckCircle;
    if (value >= thresholds.okay) return AlertTriangle;
    return TrendingDown;
  };

  const metricCards = [
    {
      title: 'Expected Return',
      value: formatPercentage(metrics.expected_return),
      icon: TrendingUp,
      color: getMetricColor(metrics.expected_return, { good: 0.12, okay: 0.08 }),
      description: 'Annual expected return',
      subtext: `${formatCurrency(total_amount * metrics.expected_return)} annually`
    },
    {
      title: 'Volatility',
      value: formatPercentage(metrics.volatility),
      icon: BarChart3,
      color: getMetricColor(metrics.volatility, { good: 0, okay: 0.15 }, true), // Lower is better
      description: 'Portfolio risk (standard deviation)',
      subtext: 'Lower is better'
    },
    {
      title: 'Sharpe Ratio',
      value: formatNumber(metrics.sharpe_ratio, 3),
      icon: Target,
      color: getMetricColor(metrics.sharpe_ratio, { good: 1.5, okay: 1.0 }),
      description: 'Risk-adjusted return',
      subtext: '>1.5 is excellent'
    },
    {
      title: 'Max Drawdown',
      value: formatPercentage(Math.abs(metrics.max_drawdown)),
      icon: Shield,
      color: getMetricColor(Math.abs(metrics.max_drawdown), { good: 0, okay: 0.15 }, true),
      description: 'Maximum potential loss',
      subtext: 'Historical worst-case scenario'
    }
  ];

  const additionalMetrics = [
    {
      label: 'Value at Risk (95%)',
      value: formatPercentage(Math.abs(metrics.var_95 || 0)),
      description: '5% chance of losing more than this in a day'
    },
    {
      label: 'Effective Assets',
      value: formatNumber(metrics.effective_assets || 0, 1),
      description: 'Diversification measure'
    },
    {
      label: 'Concentration Risk',
      value: formatPercentage(metrics.concentration_risk || 0),
      description: 'Largest single position'
    }
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="card"
      >
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h3 className="text-xl font-bold text-gray-900 mb-2">
              {strategy}
            </h3>
            <div className="flex items-center space-x-4 text-sm text-gray-600">
              <span className="flex items-center">
                <DollarSign className="h-4 w-4 mr-1" />
                {formatCurrency(total_amount)}
              </span>
              <span className="flex items-center">
                <Clock className="h-4 w-4 mr-1" />
                {new Date(created_at).toLocaleDateString()}
              </span>
              <span className="bg-blue-100 text-blue-800 px-2 py-1 rounded-full text-xs">
                {portfolio_id}
              </span>
            </div>
          </div>
          
          <div className="mt-4 sm:mt-0 flex flex-col space-y-3">
            <div className="flex items-center space-x-2">
              {optimization_details.optimization_status ? (
                <CheckCircle className="h-5 w-5 text-green-500" />
              ) : (
                <AlertTriangle className="h-5 w-5 text-yellow-500" />
              )}
              <span className="text-sm font-medium text-gray-700">
                {optimization_details.optimization_message || 'Optimization Complete'}
              </span>
            </div>
            <button
              onClick={openSaveModal}
              className="flex items-center space-x-2 btn-primary text-sm"
            >
              <Bookmark className="h-4 w-4" />
              <span>Save Portfolio</span>
            </button>
          </div>
        </div>
      </motion.div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {metricCards.map((metric, index) => {
          const Icon = metric.icon;
          return (
            <motion.div
              key={metric.title}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className="card hover:shadow-md transition-shadow"
            >
              <div className="flex items-center justify-between mb-3">
                <Icon className={`h-6 w-6 ${metric.color}`} />
                <div className={`text-2xl font-bold ${metric.color}`}>
                  {metric.value}
                </div>
              </div>
              <h4 className="font-semibold text-gray-900 mb-1">{metric.title}</h4>
              <p className="text-sm text-gray-600 mb-1">{metric.description}</p>
              <p className="text-xs text-gray-500">{metric.subtext}</p>
            </motion.div>
          );
        })}
      </div>

      {/* Additional Metrics */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="card"
      >
        <h4 className="text-lg font-semibold text-gray-900 mb-4">Additional Risk Metrics</h4>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {additionalMetrics.map((metric, index) => (
            <div key={metric.label} className="text-center">
              <div className="text-xl font-bold text-blue-600 mb-1">
                {metric.value}
              </div>
              <div className="font-medium text-gray-900 mb-1">
                {metric.label}
              </div>
              <div className="text-xs text-gray-600">
                {metric.description}
              </div>
            </div>
          ))}
        </div>
      </motion.div>

      {/* Holdings Table */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
        className="card"
      >
        <h4 className="text-lg font-semibold text-gray-900 mb-4">
          Portfolio Holdings ({holdings.length} assets)
        </h4>
        
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Asset
                </th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Weight
                </th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Amount
                </th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Expected Return
                </th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Volatility
                </th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Sharpe Ratio
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {holdings.map((holding, index) => (
                <motion.tr
                  key={holding.symbol}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.6 + index * 0.05 }}
                  className="hover:bg-gray-50"
                >
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <div className="flex-shrink-0 h-8 w-8">
                        <div className="h-8 w-8 rounded-full bg-blue-100 flex items-center justify-center">
                          <span className="text-xs font-medium text-blue-600">
                            {holding.symbol.slice(0, 2)}
                          </span>
                        </div>
                      </div>
                      <div className="ml-3">
                        <div className="text-sm font-medium text-gray-900">
                          {holding.symbol}
                        </div>
                        <div className="text-sm text-gray-500">
                          {holding.sector}
                        </div>
                      </div>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right">
                    <div className="text-sm font-medium text-gray-900">
                      {formatPercentage(holding.weight)}
                    </div>
                    <div className="w-20 bg-gray-200 rounded-full h-2 ml-auto mt-1">
                      <div
                        className="bg-blue-600 h-2 rounded-full"
                        style={{ width: `${holding.weight * 100}%` }}
                      ></div>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right text-sm text-gray-900">
                    {formatCurrency(holding.allocation_amount)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right text-sm text-gray-900">
                    {formatPercentage(holding.expected_return || 0)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right text-sm text-gray-900">
                    {formatPercentage(holding.volatility || 0)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right text-sm text-gray-900">
                    {formatNumber(holding.sharpe_ratio || 0, 2)}
                  </td>
                </motion.tr>
              ))}
            </tbody>
          </table>
        </div>
      </motion.div>

      {/* Optimization Details */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.7 }}
        className="card bg-gray-50"
      >
        <h4 className="text-lg font-semibold text-gray-900 mb-4">Optimization Details</h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div>
            <div className="font-medium text-gray-700">Constraint Level</div>
            <div className="text-gray-600 capitalize">{optimization_details.constraint_level}</div>
          </div>
          <div>
            <div className="font-medium text-gray-700">Return Method</div>
            <div className="text-gray-600">{optimization_details.return_method?.replace('_', ' ')}</div>
          </div>
          <div>
            <div className="font-medium text-gray-700">Objective</div>
            <div className="text-gray-600 capitalize">{optimization_details.objective?.replace('_', ' ')}</div>
          </div>
          <div>
            <div className="font-medium text-gray-700">Max Position</div>
            <div className="text-gray-600">{formatPercentage(optimization_details.max_position_size || 0)}</div>
          </div>
        </div>
      </motion.div>

      {/* Save Portfolio Modal */}
      <AnimatePresence>
        {showSaveModal && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50"
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="bg-white rounded-xl p-6 max-w-md w-full"
            >
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-semibold">Save Portfolio</h3>
                <button
                  onClick={() => setShowSaveModal(false)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  <X className="h-6 w-6" />
                </button>
              </div>

              {saveSuccess ? (
                <div className="text-center py-4">
                  <CheckCircle className="h-12 w-12 text-green-500 mx-auto mb-3" />
                  <h4 className="text-lg font-medium text-gray-900 mb-1">
                    Portfolio Saved!
                  </h4>
                  <p className="text-gray-600">
                    Your portfolio has been saved successfully.
                  </p>
                </div>
              ) : (
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Portfolio Name *
                    </label>
                    <input
                      type="text"
                      value={portfolioName}
                      onChange={(e) => setPortfolioName(e.target.value)}
                      className="input-field"
                      placeholder="Enter portfolio name"
                      autoFocus
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Description (Optional)
                    </label>
                    <textarea
                      value={portfolioDescription}
                      onChange={(e) => setPortfolioDescription(e.target.value)}
                      className="input-field"
                      rows={3}
                      placeholder="Add a description for this portfolio"
                    />
                  </div>

                  {saveError && (
                    <div className="bg-red-50 border border-red-200 rounded-lg p-3">
                      <p className="text-red-600 text-sm">{saveError}</p>
                    </div>
                  )}

                  <div className="bg-gray-50 rounded-lg p-3">
                    <div className="text-sm text-gray-600 space-y-1">
                      <div className="flex justify-between">
                        <span>Expected Return:</span>
                        <span className="font-medium">{formatPercentage(metrics.expected_return)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Volatility:</span>
                        <span className="font-medium">{formatPercentage(metrics.volatility)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Sharpe Ratio:</span>
                        <span className="font-medium">{formatNumber(metrics.sharpe_ratio, 2)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Total Value:</span>
                        <span className="font-medium">{formatCurrency(total_amount)}</span>
                      </div>
                    </div>
                  </div>

                  <div className="flex justify-end space-x-3">
                    <button
                      onClick={() => setShowSaveModal(false)}
                      className="btn-secondary"
                      disabled={saving}
                    >
                      Cancel
                    </button>
                    <button
                      onClick={handleSavePortfolio}
                      className="btn-primary"
                      disabled={saving || !portfolioName.trim()}
                    >
                      {saving ? (
                        <div className="flex items-center space-x-2">
                          <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                          <span>Saving...</span>
                        </div>
                      ) : (
                        'Save Portfolio'
                      )}
                    </button>
                  </div>
                </div>
              )}
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default OptimizationResults; 