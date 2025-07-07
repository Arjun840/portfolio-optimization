import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  PieChart, 
  Pie, 
  Cell, 
  ResponsiveContainer, 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend 
} from 'recharts';
import { PieChart as PieChartIcon, BarChart3 } from 'lucide-react';

const AllocationChart = ({ data }) => {
  const [chartType, setChartType] = useState('holdings'); // 'holdings' or 'clusters'

  if (!data || !data.holdings) {
    return (
      <div className="card">
        <div className="text-center py-8">
          <PieChartIcon className="h-12 w-12 text-gray-300 mx-auto mb-2" />
          <p className="text-gray-500">No allocation data available</p>
        </div>
      </div>
    );
  }

  const { holdings, cluster_allocation } = data;

  // Prepare holdings data for pie chart
  const holdingsData = holdings
    .filter(holding => holding.weight > 0.001) // Only show significant holdings
    .map((holding, index) => ({
      name: holding.symbol,
      value: holding.weight * 100,
      amount: holding.allocation_amount,
      sector: holding.sector,
      color: CHART_COLORS[index % CHART_COLORS.length]
    }))
    .sort((a, b) => b.value - a.value);

  // Prepare cluster data for bar chart
  const clusterData = Object.entries(cluster_allocation || {}).map(([clusterId, allocation]) => ({
    cluster: `Cluster ${clusterId}`,
    allocation: allocation * 100,
    color: CLUSTER_COLORS[parseInt(clusterId) % CLUSTER_COLORS.length]
  }));

  // Custom tooltip for pie chart
  const PieTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
          <p className="font-semibold text-gray-900">{data.name}</p>
          <p className="text-blue-600">
            Weight: <span className="font-medium">{data.value.toFixed(2)}%</span>
          </p>
          <p className="text-green-600">
            Amount: <span className="font-medium">${data.amount.toLocaleString()}</span>
          </p>
          <p className="text-gray-600 text-sm">Sector: {data.sector}</p>
        </div>
      );
    }
    return null;
  };

  // Custom tooltip for bar chart
  const BarTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      const data = payload[0];
      return (
        <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
          <p className="font-semibold text-gray-900">{label}</p>
          <p className="text-blue-600">
            Allocation: <span className="font-medium">{data.value.toFixed(2)}%</span>
          </p>
        </div>
      );
    }
    return null;
  };

  // Custom label for pie chart
  const renderCustomLabel = ({ cx, cy, midAngle, innerRadius, outerRadius, value, name }) => {
    if (value < 5) return null; // Don't show labels for small slices
    
    const RADIAN = Math.PI / 180;
    const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
    const x = cx + radius * Math.cos(-midAngle * RADIAN);
    const y = cy + radius * Math.sin(-midAngle * RADIAN);

    return (
      <text 
        x={x} 
        y={y} 
        fill="white" 
        textAnchor={x > cx ? 'start' : 'end'} 
        dominantBaseline="central"
        className="font-medium text-sm"
      >
        {`${name} ${value.toFixed(1)}%`}
      </text>
    );
  };

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.3 }}
      className="card"
    >
      {/* Header with toggle */}
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-gray-900">Portfolio Allocation</h3>
        
        <div className="flex bg-gray-100 rounded-lg p-1">
          <button
            onClick={() => setChartType('holdings')}
            className={`
              flex items-center px-3 py-2 rounded-md text-sm font-medium transition-colors
              ${chartType === 'holdings' 
                ? 'bg-white text-blue-600 shadow-sm' 
                : 'text-gray-600 hover:text-gray-900'
              }
            `}
          >
            <PieChartIcon className="h-4 w-4 mr-1" />
            Holdings
          </button>
          <button
            onClick={() => setChartType('clusters')}
            className={`
              flex items-center px-3 py-2 rounded-md text-sm font-medium transition-colors
              ${chartType === 'clusters' 
                ? 'bg-white text-blue-600 shadow-sm' 
                : 'text-gray-600 hover:text-gray-900'
              }
            `}
          >
            <BarChart3 className="h-4 w-4 mr-1" />
            Clusters
          </button>
        </div>
      </div>

      {chartType === 'holdings' ? (
        <div className="space-y-6">
          {/* Pie Chart */}
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={holdingsData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={renderCustomLabel}
                  outerRadius={120}
                  fill="#8884d8"
                  dataKey="value"
                  animationBegin={0}
                  animationDuration={1000}
                >
                  {holdingsData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip content={<PieTooltip />} />
              </PieChart>
            </ResponsiveContainer>
          </div>

          {/* Holdings Legend */}
          <div className="grid grid-cols-2 gap-3">
            {holdingsData.map((holding, index) => (
              <div key={holding.name} className="flex items-center">
                <div 
                  className="w-4 h-4 rounded-full mr-3 flex-shrink-0"
                  style={{ backgroundColor: holding.color }}
                ></div>
                <div className="flex-1 min-w-0">
                  <div className="font-medium text-gray-900 truncate">
                    {holding.name}
                  </div>
                  <div className="text-sm text-gray-500">
                    {holding.value.toFixed(2)}% • ${holding.amount.toLocaleString()}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div className="space-y-6">
          {/* Bar Chart */}
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={clusterData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis 
                  dataKey="cluster" 
                  tick={{ fontSize: 12 }}
                  stroke="#6b7280"
                />
                <YAxis 
                  tick={{ fontSize: 12 }}
                  stroke="#6b7280"
                  label={{ value: 'Allocation (%)', angle: -90, position: 'insideLeft' }}
                />
                <Tooltip content={<BarTooltip />} />
                <Bar 
                  dataKey="allocation" 
                  radius={[4, 4, 0, 0]}
                  animationDuration={1000}
                >
                  {clusterData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Cluster Legend */}
          <div className="space-y-3">
            {clusterData.map((cluster, index) => (
              <div key={cluster.cluster} className="flex items-center justify-between">
                <div className="flex items-center">
                  <div 
                    className="w-4 h-4 rounded-full mr-3"
                    style={{ backgroundColor: cluster.color }}
                  ></div>
                  <span className="font-medium text-gray-900">{cluster.cluster}</span>
                </div>
                <span className="text-sm text-gray-600">
                  {cluster.allocation.toFixed(2)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Summary Statistics */}
      <div className="mt-6 pt-6 border-t border-gray-200">
        <div className="grid grid-cols-3 gap-4 text-center">
          <div>
            <div className="text-2xl font-bold text-blue-600">{holdings.length}</div>
            <div className="text-sm text-gray-600">Assets</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-green-600">
              {(data.metrics?.effective_assets || 0).toFixed(1)}
            </div>
            <div className="text-sm text-gray-600">Effective Assets</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-purple-600">
              {((data.metrics?.concentration_risk || 0) * 100).toFixed(1)}%
            </div>
            <div className="text-sm text-gray-600">Max Position</div>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

// Color palettes
const CHART_COLORS = [
  '#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6',
  '#06b6d4', '#84cc16', '#f97316', '#6366f1', '#ec4899',
  '#14b8a6', '#eab308', '#8b5cf6', '#f43f5e', '#22d3ee',
  '#a3e635', '#fb7185', '#4ade80', '#fbbf24', '#c084fc'
];

const CLUSTER_COLORS = [
  '#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6'
];

export default AllocationChart; 