import React, { useState, useEffect } from 'react';
import { Plus, Trash2, BarChart3, TrendingUp, AlertCircle, Target } from 'lucide-react';
import authService from '../services/authService';

const CustomPortfolioBuilder = ({ stocks = [] }) => {
  const [portfolioHoldings, setPortfolioHoldings] = useState([]);
  const [selectedStock, setSelectedStock] = useState('');
  const [weightInput, setWeightInput] = useState('');
  const [optimizationType, setOptimizationType] = useState('improve');
  const [targetReturn, setTargetReturn] = useState('');
  const [maxPositionSize, setMaxPositionSize] = useState('30');
  const [allowNewAssets, setAllowNewAssets] = useState(true);
  const [preserveCoreHoldings, setPreserveCoreHoldings] = useState(false);
  const [coreHoldings, setCoreHoldings] = useState([]);
  const [portfolioValue, setPortfolioValue] = useState('100000');
  const [isOptimizing, setIsOptimizing] = useState(false);
  const [optimizationResult, setOptimizationResult] = useState(null);
  const [errors, setErrors] = useState({});
  const [originalPortfolio, setOriginalPortfolio] = useState([]);

  // Calculate total weight
  const totalWeight = portfolioHoldings.reduce((sum, holding) => sum + holding.weight, 0);
  const isValidPortfolio = Math.abs(totalWeight - 100) < 0.1 && portfolioHoldings.length > 0;

  // Add holding to portfolio
  const addHolding = () => {
    const weight = parseFloat(weightInput);
    if (!selectedStock || !weight || weight <= 0 || weight > 100) {
      setErrors({ add: 'Please select a stock and enter a valid weight (1-100%)' });
      return;
    }

    if (portfolioHoldings.find(h => h.symbol === selectedStock)) {
      setErrors({ add: 'Stock already in portfolio' });
      return;
    }

    if (totalWeight + weight > 100) {
      setErrors({ add: 'Total weight would exceed 100%' });
      return;
    }

    const stock = stocks.find(s => s.symbol === selectedStock);
    setPortfolioHoldings([...portfolioHoldings, {
      symbol: selectedStock,
      name: stock?.name || selectedStock,
      weight: weight,
      expected_return: stock?.expected_return || 0,
      volatility: stock?.volatility || 0,
      sharpe_ratio: stock?.sharpe_ratio || 0,
      sector: stock?.sector || 'Unknown',
      cluster: stock?.cluster || 0
    }]);

    setSelectedStock('');
    setWeightInput('');
    setErrors({});
  };

  // Remove holding from portfolio
  const removeHolding = (symbol) => {
    setPortfolioHoldings(portfolioHoldings.filter(h => h.symbol !== symbol));
    setCoreHoldings(coreHoldings.filter(s => s !== symbol));
  };

  // Update holding weight
  const updateWeight = (symbol, newWeight) => {
    const weight = parseFloat(newWeight);
    if (isNaN(weight) || weight < 0 || weight > 100) return;

    setPortfolioHoldings(portfolioHoldings.map(h => 
      h.symbol === symbol ? { ...h, weight } : h
    ));
  };

  // Normalize weights to 100%
  const normalizeWeights = () => {
    if (totalWeight === 0) return;
    
    const factor = 100 / totalWeight;
    setPortfolioHoldings(portfolioHoldings.map(h => ({
      ...h,
      weight: Math.round(h.weight * factor * 100) / 100
    })));
  };

  // Toggle core holding
  const toggleCoreHolding = (symbol) => {
    if (coreHoldings.includes(symbol)) {
      setCoreHoldings(coreHoldings.filter(s => s !== symbol));
    } else {
      setCoreHoldings([...coreHoldings, symbol]);
    }
  };

  // Optimize custom portfolio
  const optimizePortfolio = async () => {
    if (!isValidPortfolio) {
      setErrors({ optimize: 'Portfolio must have holdings totaling 100%' });
      return;
    }

    setIsOptimizing(true);
    setErrors({});
    
    // Save original portfolio before optimization
    setOriginalPortfolio([...portfolioHoldings]);

    try {
      const holdings = portfolioHoldings.map(h => ({
        symbol: h.symbol,
        weight: h.weight / 100 // Convert percentage to decimal
      }));

      const request = {
        holdings,
        optimization_type: optimizationType,
        target_return: targetReturn ? parseFloat(targetReturn) / 100 : null,
        max_position_size: parseFloat(maxPositionSize) / 100,
        constraint_level: 'enhanced',
        return_method: 'ml_enhanced',
        allow_new_assets: allowNewAssets,
        preserve_core_holdings: preserveCoreHoldings,
        core_holdings: preserveCoreHoldings ? coreHoldings : null
      };

      const response = await authService.optimizeCustomPortfolio(request, parseFloat(portfolioValue));
      setOptimizationResult(response);
      
      // Update portfolio holdings with optimized allocations
      if (response.holdings && response.holdings.length > 0) {
        const optimizedHoldings = response.holdings.map(holding => {
          const stock = stocks.find(s => s.symbol === holding.symbol);
          return {
            symbol: holding.symbol,
            name: stock?.name || holding.symbol,
            weight: Math.round(holding.weight * 100 * 100) / 100, // Convert to percentage and round
            expected_return: stock?.expected_return || 0,
            volatility: stock?.volatility || 0,
            sharpe_ratio: stock?.sharpe_ratio || 0,
            sector: stock?.sector || 'Unknown',
            cluster: stock?.cluster || 0
          };
        });
        setPortfolioHoldings(optimizedHoldings);
        
        // Update core holdings if preserve option was enabled
        if (preserveCoreHoldings) {
          const newCoreHoldings = optimizedHoldings
            .filter(h => coreHoldings.includes(h.symbol))
            .map(h => h.symbol);
          setCoreHoldings(newCoreHoldings);
        }
      }
    } catch (error) {
      console.error('Optimization error:', error);
      setErrors({ optimize: error.response?.data?.detail || 'Optimization failed' });
    } finally {
      setIsOptimizing(false);
    }
  };

  // Reset to original portfolio before optimization
  const resetToOriginal = () => {
    if (originalPortfolio.length > 0) {
      setPortfolioHoldings([...originalPortfolio]);
      setOptimizationResult(null);
    }
  };

  // Quick portfolio templates
  const applyTemplate = (template) => {
    let holdings = [];
    
    switch (template) {
      case 'balanced':
        holdings = [
          { symbol: 'SPY', weight: 40 },
          { symbol: 'QQQ', weight: 30 },
          { symbol: 'TLT', weight: 20 },
          { symbol: 'GLD', weight: 10 }
        ];
        break;
      case 'growth':
        holdings = [
          { symbol: 'QQQ', weight: 35 },
          { symbol: 'AAPL', weight: 20 },
          { symbol: 'GOOGL', weight: 15 },
          { symbol: 'MSFT', weight: 15 },
          { symbol: 'NVDA', weight: 15 }
        ];
        break;
      case 'conservative':
        holdings = [
          { symbol: 'SPY', weight: 30 },
          { symbol: 'TLT', weight: 40 },
          { symbol: 'VTI', weight: 20 },
          { symbol: 'GLD', weight: 10 }
        ];
        break;
    }

    const expandedHoldings = holdings.map(h => {
      const stock = stocks.find(s => s.symbol === h.symbol);
      return {
        ...h,
        name: stock?.name || h.symbol,
        expected_return: stock?.expected_return || 0,
        volatility: stock?.volatility || 0,
        sharpe_ratio: stock?.sharpe_ratio || 0,
        sector: stock?.sector || 'Unknown',
        cluster: stock?.cluster || 0
      };
    }).filter(h => stocks.find(s => s.symbol === h.symbol)); // Only include available stocks

    setPortfolioHoldings(expandedHoldings);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Custom Portfolio Builder</h2>
          <p className="text-gray-600 mt-1">Build your own portfolio and optimize it</p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => applyTemplate('balanced')}
            className="btn-secondary text-sm"
          >
            Balanced Template
          </button>
          <button
            onClick={() => applyTemplate('growth')}
            className="btn-secondary text-sm"
          >
            Growth Template
          </button>
          <button
            onClick={() => applyTemplate('conservative')}
            className="btn-secondary text-sm"
          >
            Conservative Template
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Portfolio Builder */}
        <div className="space-y-6">
          {/* Add Holdings */}
          <div className="card">
            <div className="card-header">
              <h3 className="font-semibold text-gray-900 flex items-center gap-2">
                <Plus className="w-5 h-5" />
                Add Holdings
              </h3>
            </div>
            <div className="card-content space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                <select
                  value={selectedStock}
                  onChange={(e) => setSelectedStock(e.target.value)}
                  className="input-field"
                >
                  <option value="">Select Stock</option>
                  {stocks
                    .filter(stock => !portfolioHoldings.find(h => h.symbol === stock.symbol))
                    .map(stock => (
                      <option key={stock.symbol} value={stock.symbol}>
                        {stock.symbol} - {(stock.expected_return * 100).toFixed(1)}% return
                      </option>
                    ))}
                </select>
                <input
                  type="number"
                  placeholder="Weight %"
                  value={weightInput}
                  onChange={(e) => setWeightInput(e.target.value)}
                  className="input-field"
                  min="0"
                  max="100"
                  step="0.1"
                />
                <button
                  onClick={addHolding}
                  className="btn-primary"
                  disabled={!selectedStock || !weightInput}
                >
                  Add
                </button>
              </div>
              {errors.add && (
                <div className="text-red-600 text-sm flex items-center gap-1">
                  <AlertCircle className="w-4 h-4" />
                  {errors.add}
                </div>
              )}
            </div>
          </div>

          {/* Current Portfolio */}
          <div className="card">
            <div className="card-header">
              <div className="flex items-center justify-between">
                <h3 className="font-semibold text-gray-900">Current Portfolio</h3>
                <div className="text-sm">
                  <span className={`font-medium ${Math.abs(totalWeight - 100) < 0.1 ? 'text-green-600' : 'text-red-600'}`}>
                    {totalWeight.toFixed(1)}%
                  </span>
                  <span className="text-gray-500 ml-1">of 100%</span>
                </div>
              </div>
            </div>
            <div className="card-content">
              {portfolioHoldings.length === 0 ? (
                <div className="text-center py-8 text-gray-500">
                  <BarChart3 className="w-12 h-12 mx-auto mb-3 opacity-50" />
                  <p>No holdings added yet</p>
                </div>
              ) : (
                <div className="space-y-3">
                  {portfolioHoldings.map((holding) => (
                    <div
                      key={holding.symbol}
                      className="flex items-center justify-between p-3 bg-gray-50 rounded-lg"
                    >
                      <div className="flex items-center gap-3">
                        <div>
                          <div className="font-medium text-gray-900">{holding.symbol}</div>
                          <div className="text-sm text-gray-500">
                            {(holding.expected_return * 100).toFixed(1)}% return, 
                            {(holding.volatility * 100).toFixed(1)}% volatility
                          </div>
                        </div>
                        {preserveCoreHoldings && (
                          <button
                            onClick={() => toggleCoreHolding(holding.symbol)}
                            className={`text-xs px-2 py-1 rounded-full ${
                              coreHoldings.includes(holding.symbol)
                                ? 'bg-blue-100 text-blue-700'
                                : 'bg-gray-200 text-gray-600'
                            }`}
                          >
                            {coreHoldings.includes(holding.symbol) ? 'Core' : 'Non-core'}
                          </button>
                        )}
                      </div>
                      <div className="flex items-center gap-2">
                        <input
                          type="number"
                          value={holding.weight}
                          onChange={(e) => updateWeight(holding.symbol, e.target.value)}
                          className="w-20 px-2 py-1 text-sm border border-gray-300 rounded"
                          min="0"
                          max="100"
                          step="0.1"
                        />
                        <span className="text-sm text-gray-500">%</span>
                        <button
                          onClick={() => removeHolding(holding.symbol)}
                          className="text-red-600 hover:text-red-700 p-1"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                  ))}
                  
                  {Math.abs(totalWeight - 100) > 0.1 && (
                    <button
                      onClick={normalizeWeights}
                      className="w-full btn-secondary text-sm"
                    >
                      Normalize to 100%
                    </button>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Optimization Settings */}
        <div className="space-y-6">
          {/* Settings */}
          <div className="card">
            <div className="card-header">
              <h3 className="font-semibold text-gray-900 flex items-center gap-2">
                <Target className="w-5 h-5" />
                Optimization Settings
              </h3>
            </div>
            <div className="card-content space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Portfolio Value
                </label>
                <input
                  type="number"
                  value={portfolioValue}
                  onChange={(e) => setPortfolioValue(e.target.value)}
                  className="input-field"
                  placeholder="100000"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Optimization Type
                </label>
                <div className="space-y-2">
                  {[
                    { value: 'improve', label: 'Improve Portfolio', desc: 'Enhance risk-return profile' },
                    { value: 'rebalance', label: 'Rebalance Only', desc: 'Keep same stocks, adjust weights' },
                    { value: 'risk_adjust', label: 'Risk Adjustment', desc: 'Optimize for target risk level' }
                  ].map(option => (
                    <label key={option.value} className="flex items-start gap-3 cursor-pointer">
                      <input
                        type="radio"
                        name="optimizationType"
                        value={option.value}
                        checked={optimizationType === option.value}
                        onChange={(e) => setOptimizationType(e.target.value)}
                        className="mt-0.5"
                      />
                      <div>
                        <div className="font-medium text-gray-900">{option.label}</div>
                        <div className="text-sm text-gray-500">{option.desc}</div>
                      </div>
                    </label>
                  ))}
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Target Return (Optional)
                </label>
                <input
                  type="number"
                  value={targetReturn}
                  onChange={(e) => setTargetReturn(e.target.value)}
                  className="input-field"
                  placeholder="15"
                  step="0.1"
                />
                <div className="text-xs text-gray-500 mt-1">Annual return percentage</div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Max Position Size
                </label>
                <input
                  type="number"
                  value={maxPositionSize}
                  onChange={(e) => setMaxPositionSize(e.target.value)}
                  className="input-field"
                  min="1"
                  max="100"
                  step="1"
                />
                <div className="text-xs text-gray-500 mt-1">Maximum weight per asset (%)</div>
              </div>

              <div className="space-y-3">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={allowNewAssets}
                    onChange={(e) => setAllowNewAssets(e.target.checked)}
                    className="rounded"
                  />
                  <span className="text-sm font-medium text-gray-700">Allow new assets</span>
                </label>

                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={preserveCoreHoldings}
                    onChange={(e) => setPreserveCoreHoldings(e.target.checked)}
                    className="rounded"
                  />
                  <span className="text-sm font-medium text-gray-700">Preserve core holdings</span>
                </label>
              </div>

              <button
                onClick={optimizePortfolio}
                disabled={!isValidPortfolio || isOptimizing}
                className="w-full btn-primary flex items-center justify-center gap-2"
              >
                {isOptimizing ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                    Optimizing...
                  </>
                ) : (
                  <>
                    <TrendingUp className="w-5 h-5" />
                    Optimize Portfolio
                  </>
                )}
              </button>

              {errors.optimize && (
                <div className="text-red-600 text-sm flex items-center gap-1">
                  <AlertCircle className="w-4 h-4" />
                  {errors.optimize}
                </div>
              )}
            </div>
          </div>

          {/* Optimization Results */}
          {optimizationResult && (
            <div className="card border-green-200 bg-green-50">
              <div className="card-header">
                <h3 className="font-semibold text-gray-900 flex items-center gap-2">
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                  Portfolio Optimized Successfully
                </h3>
                <p className="text-sm text-green-700 mt-1">
                  Your portfolio allocations have been updated with the optimized weights.
                </p>
              </div>
              <div className="card-content">
                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div className="text-center p-3 bg-green-50 rounded-lg">
                    <div className="text-lg font-bold text-green-700">
                      {(optimizationResult.metrics.expected_return * 100).toFixed(1)}%
                    </div>
                    <div className="text-sm text-green-600">Expected Return</div>
                  </div>
                  <div className="text-center p-3 bg-blue-50 rounded-lg">
                    <div className="text-lg font-bold text-blue-700">
                      {optimizationResult.metrics.sharpe_ratio.toFixed(2)}
                    </div>
                    <div className="text-sm text-blue-600">Sharpe Ratio</div>
                  </div>
                </div>

                {optimizationResult.optimization_details.improvement_metrics && (
                  <div className="text-sm text-gray-600 mb-3">
                    <strong>Improvement:</strong> {optimizationResult.optimization_details.improvement_metrics.improvement_pct.toFixed(1)}% better Sharpe ratio
                  </div>
                )}

                <div className="text-sm mb-4">
                  <strong>Strategy:</strong> {optimizationResult.strategy}
                  <br />
                  <strong>Total Value:</strong> ${optimizationResult.total_amount.toLocaleString()}
                  <br />
                  <strong>Active Assets:</strong> {optimizationResult.holdings.length}
                </div>
                
                {originalPortfolio.length > 0 && (
                  <button
                    onClick={resetToOriginal}
                    className="btn-secondary text-sm"
                  >
                    Reset to Original Portfolio
                  </button>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default CustomPortfolioBuilder; 