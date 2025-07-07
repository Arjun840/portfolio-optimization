import React from 'react';
import { motion } from 'framer-motion';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  AreaChart,
  Area
} from 'recharts';
import { TrendingUp, BarChart3 } from 'lucide-react';

const PortfolioChart = ({ data }) => {
  if (!data || !data.metrics) {
    return (
      <div className="card">
        <div className="text-center py-8">
          <BarChart3 className="h-12 w-12 text-gray-300 mx-auto mb-2" />
          <p className="text-gray-500">No performance data available</p>
        </div>
      </div>
    );
  }

  const { metrics } = data;

  // Generate sample performance data (in a real app, this would come from the API)
  const generatePerformanceData = () => {
    const months = 12;
    const startValue = 100;
    const monthlyReturn = metrics.expected_return / 12;
    const monthlyVolatility = metrics.volatility / Math.sqrt(12);
    
    const performanceData = [];
    let portfolioValue = startValue;
    let benchmarkValue = startValue;
    
    for (let i = 0; i <= months; i++) {
      const month = new Date();
      month.setMonth(month.getMonth() - (months - i));
      
      // Simulate some randomness for demonstration
      const randomFactor = (Math.random() - 0.5) * monthlyVolatility * 2;
      const portfolioReturn = monthlyReturn + randomFactor;
      const benchmarkReturn = 0.08 / 12 + (Math.random() - 0.5) * 0.02; // S&P 500 approximation
      
      portfolioValue *= (1 + portfolioReturn);
      benchmarkValue *= (1 + benchmarkReturn);
      
      performanceData.push({
        month: month.toLocaleDateString('en-US', { month: 'short', year: '2-digit' }),
        portfolio: portfolioValue,
        benchmark: benchmarkValue,
        portfolioReturn: ((portfolioValue - startValue) / startValue) * 100,
        benchmarkReturn: ((benchmarkValue - startValue) / startValue) * 100
      });
    }
    
    return performanceData;
  };

  const performanceData = generatePerformanceData();

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
          <p className="font-semibold text-gray-900 mb-2">{label}</p>
          {payload.map((entry, index) => (
            <div key={index} className="flex items-center justify-between min-w-0">
              <span className="text-sm" style={{ color: entry.color }}>
                {entry.name === 'portfolio' ? 'Your Portfolio' : 'Benchmark (S&P 500)'}:
              </span>
              <span className="font-medium ml-2" style={{ color: entry.color }}>
                {entry.value.toFixed(1)}%
              </span>
            </div>
          ))}
        </div>
      );
    }
    return null;
  };

  const lastDataPoint = performanceData[performanceData.length - 1];
  const portfolioPerformance = lastDataPoint?.portfolioReturn || 0;
  const benchmarkPerformance = lastDataPoint?.benchmarkReturn || 0;
  const outperformance = portfolioPerformance - benchmarkPerformance;

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.3 }}
      className="card"
    >
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-gray-900">Performance Comparison</h3>
        <div className="flex items-center text-sm text-gray-600">
          <TrendingUp className="h-4 w-4 mr-1" />
          12 Month Projection
        </div>
      </div>

      {/* Performance Summary */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        <div className="text-center">
          <div className={`text-2xl font-bold ${portfolioPerformance >= 0 ? 'text-green-600' : 'text-red-600'}`}>
            {portfolioPerformance >= 0 ? '+' : ''}{portfolioPerformance.toFixed(1)}%
          </div>
          <div className="text-sm text-gray-600">Your Portfolio</div>
        </div>
        <div className="text-center">
          <div className={`text-2xl font-bold ${benchmarkPerformance >= 0 ? 'text-green-600' : 'text-red-600'}`}>
            {benchmarkPerformance >= 0 ? '+' : ''}{benchmarkPerformance.toFixed(1)}%
          </div>
          <div className="text-sm text-gray-600">S&P 500</div>
        </div>
        <div className="text-center">
          <div className={`text-2xl font-bold ${outperformance >= 0 ? 'text-blue-600' : 'text-orange-600'}`}>
            {outperformance >= 0 ? '+' : ''}{outperformance.toFixed(1)}%
          </div>
          <div className="text-sm text-gray-600">Outperformance</div>
        </div>
      </div>

      {/* Chart */}
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={performanceData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
            <defs>
              <linearGradient id="portfolioGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
              </linearGradient>
              <linearGradient id="benchmarkGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#6b7280" stopOpacity={0.3}/>
                <stop offset="95%" stopColor="#6b7280" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis 
              dataKey="month" 
              tick={{ fontSize: 12 }}
              stroke="#6b7280"
            />
            <YAxis 
              tick={{ fontSize: 12 }}
              stroke="#6b7280"
              label={{ value: 'Return (%)', angle: -90, position: 'insideLeft' }}
              domain={['dataMin - 2', 'dataMax + 2']}
            />
            <Tooltip content={<CustomTooltip />} />
            <Area
              type="monotone"
              dataKey="benchmarkReturn"
              stroke="#6b7280"
              strokeWidth={2}
              fill="url(#benchmarkGradient)"
              name="benchmark"
            />
            <Area
              type="monotone"
              dataKey="portfolioReturn"
              stroke="#3b82f6"
              strokeWidth={3}
              fill="url(#portfolioGradient)"
              name="portfolio"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Legend */}
      <div className="flex items-center justify-center space-x-6 mt-4">
        <div className="flex items-center">
          <div className="w-4 h-4 bg-blue-500 rounded-full mr-2"></div>
          <span className="text-sm text-gray-700">Your Portfolio</span>
        </div>
        <div className="flex items-center">
          <div className="w-4 h-4 bg-gray-500 rounded-full mr-2"></div>
          <span className="text-sm text-gray-700">S&P 500 Benchmark</span>
        </div>
      </div>

      {/* Risk Metrics */}
      <div className="mt-6 pt-6 border-t border-gray-200">
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <div className="font-medium text-gray-700">Sharpe Ratio</div>
            <div className="text-blue-600 font-semibold">{metrics.sharpe_ratio.toFixed(3)}</div>
          </div>
          <div>
            <div className="font-medium text-gray-700">Max Drawdown</div>
            <div className="text-red-600 font-semibold">{(Math.abs(metrics.max_drawdown) * 100).toFixed(1)}%</div>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default PortfolioChart; 