import mongoose from 'mongoose';

const AlertSchema = new mongoose.Schema({
  district: {
    type: String,
    required: true,
  },
  date: {
    type: Date,
    required: true,
  },
  risk_level: {
    type: String,
    enum: ['High Risk', 'Critical Risk'],
    default: null,
  },
  confidence: {
    type: String,
    default: null,
  },
  reason: {
    type: String,
    default: null,
  },
  rainfall_avg: {
    type: Number,
    default: null,
  },
  rolling_zscore: {
    type: Number,
    default: null,
  },
  acknowledged: {
    type: Boolean,
    default: false,
  },
  acknowledged_at: {
    type: Date,
    default: null,
  },
  created_at: {
    type: Date,
    default: Date.now,
  },
});

// Compound index on (risk_level, date DESC)
AlertSchema.index({ risk_level: 1, date: -1 });

// Index on acknowledged
AlertSchema.index({ acknowledged: 1 });

export default mongoose.model('Alert', AlertSchema);
