import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Bookmark, 
  Trash2, 
  PencilLine, 
  Eye, 
  Plus,
  Calendar,
  TrendingUp,
  Shield,
  BarChart3,
  X
} from 'lucide-react';
import authService from '../services/authService';

const SavedPortfolios = () => {
  const [savedPortfolios, setSavedPortfolios] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedPortfolio, setSelectedPortfolio] = useState(null);
  const [showSaveModal, setShowSaveModal] = useState(false);
  const [editingPortfolio, setEditingPortfolio] = useState(null);
  const [portfolioName, setPortfolioName] = useState('');
  const [portfolioDescription, setPortfolioDescription] = useState('');

  useEffect(() => {
    loadSavedPortfolios();
  }, []);

  const loadSavedPortfolios = async () => {
    try {
      setLoading(true);
      const result = await authService.getSavedPortfolios();
      if (result.success) {
        setSavedPortfolios(result.data);
      } else {
        setError(result.error);
      }
    } catch (err) {
      setError('Failed to load saved portfolios');
    } finally {
      setLoading(false);
    }
  };

  const handleDeletePortfolio = async (savedId) => {
    if (!window.confirm('Are you sure you want to delete this portfolio?')) {
      return;
    }

    try {
      const result = await authService.deleteSavedPortfolio(savedId);
      if (result.success) {
        setSavedPortfolios(prev => prev.filter(p => p.saved_id !== savedId));
      } else {
        setError(result.error);
      }
    } catch (err) {
      setError('Failed to delete portfolio');
    }
  };

  const handleEditPortfolio = async () => {
    try {
      const result = await authService.updateSavedPortfolio(
        editingPortfolio.saved_id,
        portfolioName,
        portfolioDescription
      );
      if (result.success) {
        setSavedPortfolios(prev => 
          prev.map(p => p.saved_id === editingPortfolio.saved_id ? result.data : p)
        );
        setEditingPortfolio(null);
        setPortfolioName('');
        setPortfolioDescription('');
      } else {
        setError(result.error);
      }
    } catch (err) {
      setError('Failed to update portfolio');
    }
  };

  const openEditModal = (portfolio) => {
    setEditingPortfolio(portfolio);
    setPortfolioName(portfolio.portfolio_name);
    setPortfolioDescription(portfolio.description || '');
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    });
  };

  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(value);
  };

  const formatPercentage = (value) => {
    return `${(value * 100).toFixed(1)}%`;
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        <span className="ml-2 text-gray-600">Loading saved portfolios...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <p className="text-red-600">{error}</p>
        <button 
          onClick={loadSavedPortfolios}
          className="mt-2 btn-primary text-sm"
        >
          Try Again
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Saved Portfolios</h2>
          <p className="text-gray-600">Manage your saved investment portfolios</p>
        </div>
        {savedPortfolios.length > 0 && (
          <span className="bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm font-medium">
            {savedPortfolios.length} Portfolio{savedPortfolios.length !== 1 ? 's' : ''}
          </span>
        )}
      </div>

      {savedPortfolios.length === 0 ? (
        <div className="text-center py-12">
          <Bookmark className="mx-auto h-12 w-12 text-gray-400" />
          <h3 className="mt-4 text-lg font-medium text-gray-900">No saved portfolios</h3>
          <p className="mt-2 text-gray-600">
            Save portfolios from the optimization results to view them here.
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {savedPortfolios.map((portfolio) => (
            <motion.div
              key={portfolio.saved_id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="card hover:shadow-lg transition-shadow duration-200"
            >
              <div className="flex justify-between items-start mb-4">
                <div className="flex-1">
                  <h3 className="font-semibold text-gray-900 truncate">
                    {portfolio.portfolio_name}
                  </h3>
                  <p className="text-sm text-gray-600 mt-1">{portfolio.strategy}</p>
                </div>
                <div className="flex space-x-1 ml-2">
                  <button
                    onClick={() => openEditModal(portfolio)}
                    className="p-1 text-gray-400 hover:text-blue-600 transition-colors"
                    title="Edit portfolio"
                  >
                    <PencilLine className="h-4 w-4" />
                  </button>
                  <button
                    onClick={() => handleDeletePortfolio(portfolio.saved_id)}
                    className="p-1 text-gray-400 hover:text-red-600 transition-colors"
                    title="Delete portfolio"
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                </div>
              </div>

              {portfolio.description && (
                <p className="text-sm text-gray-600 mb-4 line-clamp-2">
                  {portfolio.description}
                </p>
              )}

              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Total Value</span>
                  <span className="font-semibold">{formatCurrency(portfolio.total_amount)}</span>
                </div>

                <div className="grid grid-cols-3 gap-3 text-center">
                  <div className="bg-green-50 rounded-lg p-3">
                    <TrendingUp className="h-5 w-5 text-green-600 mx-auto mb-1" />
                    <p className="text-xs text-gray-600">Return</p>
                    <p className="font-semibold text-green-700">
                      {formatPercentage(portfolio.expected_return)}
                    </p>
                  </div>
                  
                  <div className="bg-blue-50 rounded-lg p-3">
                    <Shield className="h-5 w-5 text-blue-600 mx-auto mb-1" />
                    <p className="text-xs text-gray-600">Risk</p>
                    <p className="font-semibold text-blue-700">
                      {formatPercentage(portfolio.volatility)}
                    </p>
                  </div>
                  
                  <div className="bg-purple-50 rounded-lg p-3">
                    <BarChart3 className="h-5 w-5 text-purple-600 mx-auto mb-1" />
                    <p className="text-xs text-gray-600">Sharpe</p>
                    <p className="font-semibold text-purple-700">
                      {portfolio.sharpe_ratio.toFixed(2)}
                    </p>
                  </div>
                </div>

                <div className="flex items-center justify-between text-xs text-gray-500 pt-2 border-t border-gray-100">
                  <div className="flex items-center">
                    <Calendar className="h-4 w-4 mr-1" />
                    {formatDate(portfolio.created_at)}
                  </div>
                  <button
                    onClick={() => setSelectedPortfolio(portfolio)}
                    className="flex items-center text-blue-600 hover:text-blue-700 font-medium"
                  >
                    <Eye className="h-4 w-4 mr-1" />
                    View
                  </button>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      )}

      {/* Edit Portfolio Modal */}
      <AnimatePresence>
        {editingPortfolio && (
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
                <h3 className="text-lg font-semibold">Edit Portfolio</h3>
                <button
                  onClick={() => setEditingPortfolio(null)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  <X className="h-6 w-6" />
                </button>
              </div>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Portfolio Name
                  </label>
                  <input
                    type="text"
                    value={portfolioName}
                    onChange={(e) => setPortfolioName(e.target.value)}
                    className="input-field"
                    placeholder="Enter portfolio name"
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
              </div>

              <div className="flex justify-end space-x-3 mt-6">
                <button
                  onClick={() => setEditingPortfolio(null)}
                  className="btn-secondary"
                >
                  Cancel
                </button>
                <button
                  onClick={handleEditPortfolio}
                  className="btn-primary"
                  disabled={!portfolioName.trim()}
                >
                  Save Changes
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default SavedPortfolios; 