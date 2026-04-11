from django.shortcuts import render
import requests

def index(request):
    context = {
        'prediction': None,
        'accuracy': None,
        'error': None,
        'color': 'primary',  # Default color
        'data': {} # to pre-fill the form
    }
    
    if request.method == 'POST':
        # Retrieve data from form
        try:
            data = {
                "age": int(request.POST.get("age")),
                "bmi": float(request.POST.get("bmi")),
                "hemoglobin": float(request.POST.get("hemoglobin")),
                "blood_pressure": int(request.POST.get("blood_pressure")),
                "sugar_level": int(request.POST.get("sugar_level")),
                "protein_intake": int(request.POST.get("protein_intake"))
            }
            context['data'] = data
            
            # Send POST to FastAPI
            fastapi_url = "http://127.0.0.1:8001/predict"
            response = requests.post(fastapi_url, json=data, timeout=5)
            
            if response.status_code == 200:
                result = response.json()
                context['prediction'] = result.get('prediction')
                context['accuracy'] = result.get('accuracy')
                
                # Determine color alert
                if context['prediction'] == 'High':
                    context['color'] = 'danger'
                elif context['prediction'] == 'Medium':
                    context['color'] = 'warning'
                else: 
                    context['color'] = 'success'
                    
            else:
                context['error'] = f"FastAPI Error: {response.json().get('detail', 'Unknown Error')}"
                
        except ValueError as e:
            context['error'] = f"Invalid input data: Please ensure all fields are numeric."
        except Exception as e:
            context['error'] = f"Error connecting to backend: {str(e)}"
            
    return render(request, 'index.html', context)
