import os
import random
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def create_synthetic_data(num_samples=300):
    np.random.seed(42)
    random.seed(42)
    
    data = []
    for _ in range(num_samples):
        age = random.randint(18, 40)
        bmi = round(random.uniform(18, 32), 1)
        hemoglobin = round(random.uniform(8, 13), 1)
        blood_pressure = random.randint(90, 160)
        sugar_level = random.randint(70, 180)
        protein_intake = random.randint(30, 70)
        
        # Simple heuristic to determine risk
        risk_score = 0
        if bmi > 28 or bmi < 19: risk_score += 1
        if hemoglobin < 10: risk_score += 1
        if blood_pressure > 140 or blood_pressure < 100: risk_score += 1
        if sugar_level > 140: risk_score += 1
        if protein_intake < 45: risk_score += 1
        
        if risk_score >= 3:
            risk = 'High'
        elif risk_score == 2:
            risk = 'Medium'
        else:
            risk = 'Low'
            
        data.append([age, bmi, hemoglobin, blood_pressure, sugar_level, protein_intake, risk])
        
    df = pd.DataFrame(data, columns=['Age', 'BMI', 'Hemoglobin', 'BloodPressure', 'SugarLevel', 'ProteinIntake', 'Risk'])
    return df

def main():
    print("Generating synthetic dataset...")
    df = create_synthetic_data(300)
    
    # Save dataset
    dataset_dir = r"d:\resume rag bot\dataset"
    os.makedirs(dataset_dir, exist_ok=True)
    dataset_path = os.path.join(dataset_dir, 'data.csv')
    df.to_csv(dataset_path, index=False)
    print(f"Dataset saved to {dataset_path}")
    
    # Generate Chart
    print("Generating Risk distribution chart...")
    static_dir = r"d:\resume rag bot\static"
    os.makedirs(static_dir, exist_ok=True)
    chart_path = os.path.join(static_dir, 'chart.png')
    
    risk_counts = df['Risk'].value_counts()
    plt.figure(figsize=(8, 5))
    bars = plt.bar(risk_counts.index, risk_counts.values, color=['green', 'orange', 'red'])
    plt.title('Distribution of Pregnancy Nutrition Risk Levels')
    plt.xlabel('Risk Level')
    plt.ylabel('Number of Cases')
    plt.savefig(chart_path)
    plt.close()
    print(f"Chart saved to {chart_path}")
    
    # Train Model
    print("Training RandomForest model...")
    X = df[['Age', 'BMI', 'Hemoglobin', 'BloodPressure', 'SugarLevel', 'ProteinIntake']]
    y = df['Risk']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    acc_percentage = round(acc * 100, 2)
    print(f"Model Accuracy: {acc_percentage}%")
    
    # Save model and accuracy
    model_dir = r"d:\resume rag bot\model"
    model_path = os.path.join(model_dir, 'model.pkl')
    accuracy_path = os.path.join(model_dir, 'accuracy.pkl')
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
        
    with open(accuracy_path, 'wb') as f:
        pickle.dump(acc_percentage, f)
        
    print(f"Model saved to {model_path}")
    print(f"Accuracy saved to {accuracy_path}")

if __name__ == '__main__':
    main()
