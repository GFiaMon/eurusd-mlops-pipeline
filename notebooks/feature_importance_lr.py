# Feature importance analysis for Linear Regression
# Adapted from Random Forest feature importance code

print("\n🎯 Feature Importance Analysis (Linear Regression)")

# Linear Regression uses coefficients instead of feature_importances_
if hasattr(model, 'coef_'):
    # Get absolute values of coefficients as importance measure
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Coefficient': model.coef_,
        'Abs_Coefficient': np.abs(model.coef_)
    }).sort_values('Abs_Coefficient', ascending=False).head(15)
    
    print("\nTop 15 Most Important Features (by absolute coefficient):")
    for idx, row in feature_importance.iterrows():
        print(f"  {row['Feature']:25}: {row['Coefficient']:10.4f} (|{row['Abs_Coefficient']:.4f}|)")
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    colors = ['green' if x > 0 else 'red' for x in feature_importance['Coefficient']]
    plt.barh(feature_importance['Feature'], feature_importance['Coefficient'], color=colors)
    plt.xlabel('Coefficient Value')
    plt.title('Top 15 Feature Importances (Linear Regression)\nGreen=Positive, Red=Negative')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    # Save figure
    feature_importance_path = os.path.join(FIGURES, 'ml_feature_importance_lr.png')
    plt.savefig(feature_importance_path, dpi=100, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ Feature importance plot saved to: {feature_importance_path}")
else:
    print("⚠️ Feature importance not available for Linear Regression model")
