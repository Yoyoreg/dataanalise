from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import RegressionModel
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings("ignore")
app = Flask(__name__)

def execute_dual_path_engine(m_list, forecast_days=7):
    total_data = np.array(m_list, dtype=float)
    total_n = len(total_data)
    
    # 1. تهيئة التواريخ المتوافقة مع محرك Darts تلقائياً
    date_range = pd.date_range(start='2024-01-01', periods=total_n, freq='D')
    series_all = TimeSeries.from_series(pd.Series(total_data, index=date_range))
    
    # 🌟 المسار الأول: التحليل الاستراتيجي بعيد المدى (Macro Model)
    # يدرس التاريخ الطويل بالكامل (حتى 1000 يوم) وينظر للخلف بمقدار 60 يوماً لالتقاط الرواتب والمناسبات الدورية
    macro_lags = min(60, total_n - forecast_days - 1)
    macro_model = RegressionModel(
        model=RandomForestRegressor(n_estimators=150, random_state=42),
        lags=macro_lags, 
        output_chunk_length=forecast_days
    )
    macro_model.fit(series_all)
    macro_pred = macro_model.predict(n=forecast_days, series=series_all).values().flatten()
    
    # 🌟 المسار الثاني: التحليل التكتيكي قصير المدى (Micro Model)
    # يقتطع آخر 30 يوماً فقط من السلسلة ويركز على الذاكرة اللحظية (7 أيام) للتأقلم السريع جداً مع الطوارئ
    micro_series = series_all[-30:]
    micro_model = RegressionModel(
        model=RandomForestRegressor(n_estimators=100, random_state=42),
        lags=7,  # 🎯 التعديل الذهبي الخاص بك: 7 أيام لتغطية دورة الأسبوع بالكامل
        output_chunk_length=forecast_days
    )
    micro_model.fit(micro_series)
    micro_pred = micro_model.predict(n=forecast_days, series=micro_series).values().flatten()
    
    # 🌟 طبقة الدمج الهجين الموزون (60% للرؤية البعيدة المستقرة + 40% للتأقلم اللحظي السريع)
    final_hybrid_forecast = (0.60 * macro_pred) + (0.40 * micro_pred)
    
    # تقريب النتائج لأقرب رقمين عشريين لسهولة العرض في الهاتف
    return np.round(final_hybrid_forecast, 2).tolist()

# 📬 بوابة استقبال طلبات الهاتف (API Endpoint)
@app.route('/predict', methods=['POST'])
def predict_endpoint():
    try:
        # استقبال بيانات JSON القادمة من الهاتف
        content = request.get_json()
        
        if not content or 'data' not in content:
            return jsonify({
                "status": "error",
                "message": "❌ خطأ في الإدخال: يجب إرسال حقل 'data' محتوياً على مصفوفة النفقات التاريخية."
            }), 400
            
        expenses_history = content['data']
        predict_days = content.get('predict_days', 7) # الافتراضي هو تنبؤ للأسبوع القادم كاملاً
        
        if len(expenses_history) < 15:
            return jsonify({
                "status": "error",
                "message": "❌ السلسلة قصيرة جداً للتأقلم، يرجى إرسال سجل يحتوي على 15 إدخالاً على الأقل."
            }), 400

        # تشغيل المحرك المزدوج
        forecast_results = execute_dual_path_engine(expenses_history, forecast_days=predict_days)
        
        # 🎯 صياغة الإخراج النظيف ليكون جاهزاً للاستهلاك البرمجي في الهاتف
        return jsonify({
            "status": "success",
            "engine": "Darts Dual-Path (Macro 60d + Micro 7d)",
            "input_total_days": len(expenses_history),
            "predicted_days": predict_days,
            "predictions": forecast_results
        }), 200
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"💥 حدث خطأ داخلي في السيرفر: {str(e)}"
        }), 500

if __name__ == '__main__':
    # تشغيل السيرفر على المنفذ 5000 استقبالاً لكافة الإشارات الخارجية
    app.run(host='0.0.0.0', port=5000)
