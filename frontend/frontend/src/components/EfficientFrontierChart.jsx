import React from 'react';
import { motion } from 'framer-motion';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { TrendingUp, Target, Shield } from 'lucide-react';

const EfficientFrontierChart = ({ data }) => {
  if (!data || !data.returns || !data.volatilities || !data.sharpe_ratios || data.returns.length === 0) {
    return (
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.3 }}
        className="card"
      >
        <h3 className="text-lg font-semibold text-gray-900 mb-6">Efficient Frontier</h3>
        <div className="text-center py-12">
          <TrendingUp className="h-16 w-16 text-gray-300 mx-auto mb-4" />
          <h4 className="text-lg font-medium text-gray-900 mb-2">No Efficient Frontier Data</h4>
          <p className="text-gray-500">Click "Load Efficient Frontier" to generate the analysis</p>
        </div>
      </motion.div>
    );
  }

  // Transform API data for the scatter chart
  const chartData = data.returns.map((returnValue, index) => ({
    risk: data.volatilities[index] * 100, // Convert to percentage
    return: returnValue * 100, // Convert to percentage
    sharpeRatio: data.sharpe_ratios[index],
    portfolioId: index,
    label: `Portfolio ${index + 1}`,
    isOptimal: false, // Will be set below for max Sharpe
    isMinRisk: false, // Will be set below for min risk
  }));

  // Find key portfolios
  const maxSharpeIndex = data.sharpe_ratios.indexOf(Math.max(...data.sharpe_ratios));
  const minRiskIndex = data.volatilities.indexOf(Math.min(...data.volatilities));
  
  // Mark special portfolios
  if (chartData[maxSharpeIndex]) chartData[maxSharpeIndex].isOptimal = true;
  if (chartData[minRiskIndex]) chartData[minRiskIndex].isMinRisk = true;

  const maxSharpePortfolio = chartData[maxSharpeIndex];
  const minRiskPortfolio = chartData[minRiskIndex];

  // Custom tooltip
  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-white p-4 border border-gray-200 rounded-lg shadow-lg">
          <div className="font-semibold text-gray-900 mb-2">{data.label}</div>
          <div className="space-y-1 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-600">Expected Return:</span>
              <span className="font-medium text-green-600">{data.return.toFixed(2)}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Risk (Volatility):</span>
              <span className="font-medium text-orange-600">{data.risk.toFixed(2)}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Sharpe Ratio:</span>
              <span className="font-medium text-blue-600">{data.sharpeRatio.toFixed(3)}</span>
            </div>
            {data.isOptimal && (
              <div className="mt-2 text-xs text-blue-600 font-medium">
                ⭐ Optimal Sharpe Ratio
              </div>
            )}
            {data.isMinRisk && (
              <div className="mt-2 text-xs text-green-600 font-medium">
                🛡️ Minimum Risk
              </div>
            )}
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.3 }}
      className="card"
    >
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-gray-900">Efficient Frontier</h3>
        <div className="flex items-center text-sm text-gray-600">
          <TrendingUp className="h-4 w-4 mr-1" />
          {chartData.length} Optimal Portfolios
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="bg-blue-50 rounded-lg p-4">
          <div className="flex items-center mb-2">
            <Target className="h-5 w-5 text-blue-600 mr-2" />
            <span className="text-sm font-medium text-blue-900">Best Sharpe Ratio</span>
          </div>
          <div className="text-2xl font-bold text-blue-600">
            {maxSharpePortfolio.sharpeRatio.toFixed(3)}
          </div>
          <div className="text-sm text-blue-700">
            {maxSharpePortfolio.return.toFixed(1)}% return, {maxSharpePortfolio.risk.toFixed(1)}% risk
          </div>
        </div>
        
        <div className="bg-green-50 rounded-lg p-4">
          <div className="flex items-center mb-2">
            <Shield className="h-5 w-5 text-green-600 mr-2" />
            <span className="text-sm font-medium text-green-900">Minimum Risk</span>
          </div>
          <div className="text-2xl font-bold text-green-600">
            {minRiskPortfolio.risk.toFixed(1)}%
          </div>
          <div className="text-sm text-green-700">
            {minRiskPortfolio.return.toFixed(1)}% return, Sharpe {minRiskPortfolio.sharpeRatio.toFixed(3)}
          </div>
        </div>
      </div>

      {/* Chart */}
      <div className="h-96">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart
            margin={{ top: 20, right: 30, bottom: 20, left: 20 }}
            data={chartData}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis
              type="number"
              dataKey="risk"
              domain={['dataMin - 1', 'dataMax + 1']}
              tick={{ fontSize: 12 }}
              stroke="#6b7280"
              label={{ value: 'Risk (Volatility %)', position: 'insideBottom', offset: -10 }}
            />
            <YAxis
              type="number"
              dataKey="return"
              domain={['dataMin - 2', 'dataMax + 2']}
              tick={{ fontSize: 12 }}
              stroke="#6b7280"
              label={{ value: 'Expected Return (%)', angle: -90, position: 'insideLeft' }}
            />
            <Tooltip content={<CustomTooltip />} />
            
            {/* Regular portfolios */}
            <Scatter
              dataKey="return"
              fill="#3b82f6"
              fillOpacity={0.6}
              stroke="#1d4ed8"
              strokeWidth={1}
              r={6}
            />
            
            {/* Highlight optimal Sharpe ratio point */}
            <Scatter
              data={[maxSharpePortfolio]}
              dataKey="return"
              fill="#10b981"
              stroke="#047857"
              strokeWidth={3}
              r={10}
            />
            
            {/* Highlight minimum risk point */}
            <Scatter
              data={[minRiskPortfolio]}
              dataKey="return"
              fill="#f59e0b"
              stroke="#d97706"
              strokeWidth={3}
              r={10}
            />
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      {/* Legend */}
      <div className="flex items-center justify-center space-x-6 mt-4 text-sm">
        <div className="flex items-center">
          <div className="w-4 h-4 bg-blue-500 rounded-full mr-2"></div>
          <span className="text-gray-700">Efficient Portfolios</span>
        </div>
        <div className="flex items-center">
          <div className="w-4 h-4 bg-green-500 rounded-full mr-2"></div>
          <span className="text-gray-700">Max Sharpe Ratio</span>
        </div>
        <div className="flex items-center">
          <div className="w-4 h-4 bg-yellow-500 rounded-full mr-2"></div>
          <span className="text-gray-700">Min Risk</span>
        </div>
      </div>

      {/* Explanation */}
      <div className="mt-6 p-4 bg-gray-50 rounded-lg">
        <h4 className="font-medium text-gray-900 mb-2">Understanding the Efficient Frontier</h4>
        <p className="text-sm text-gray-600">
          Each point represents an optimal portfolio at different risk levels. The curve shows the best possible 
          return for each level of risk. Portfolios below this curve are suboptimal - you could get better 
          returns for the same risk. The green point shows the portfolio with the best risk-adjusted returns 
          (highest Sharpe ratio).
        </p>
      </div>
    </motion.div>
  );
};

export default EfficientFrontierChart; 