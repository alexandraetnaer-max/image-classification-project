"""
Generate architecture diagram
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

fig, ax = plt.subplots(figsize=(16, 12))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

# Title
ax.text(5, 11.5, 'Fashion Product Classification System Architecture', 
        ha='center', fontsize=20, fontweight='bold')

# Colors
cloud_color = '#E8F4F8'
api_color = '#FFE5B4'
model_color = '#D4E6F1'
batch_color = '#E8DAEF'
data_color = '#D5F4E6'

# 1. User/Client Layer
user_box = FancyBboxPatch((0.5, 10), 9, 0.8, 
                          boxstyle="round,pad=0.1", 
                          facecolor='#F0F0F0', 
                          edgecolor='black', linewidth=2)
ax.add_patch(user_box)
ax.text(5, 10.4, 'USER / CLIENT\n(Web Browser, curl, Applications)', 
        ha='center', va='center', fontsize=12, fontweight='bold')

# Arrow from user to cloud
arrow1 = FancyArrowPatch((5, 10), (5, 9),
                        arrowstyle='->', mutation_scale=30, 
                        linewidth=2, color='black')
ax.add_patch(arrow1)
ax.text(5.3, 9.5, 'HTTPS\nRequests', fontsize=9)

# 2. Google Cloud Run Layer
cloud_box = FancyBboxPatch((0.5, 6.5), 9, 2.3,
                          boxstyle="round,pad=0.1",
                          facecolor=cloud_color,
                          edgecolor='#3498DB', linewidth=3)
ax.add_patch(cloud_box)
ax.text(5, 8.6, 'GOOGLE CLOUD RUN', 
        ha='center', fontsize=14, fontweight='bold', color='#2874A6')

# Flask API
api_box = FancyBboxPatch((1, 7.8), 8, 0.6,
                        boxstyle="round,pad=0.05",
                        facecolor=api_color,
                        edgecolor='black', linewidth=2)
ax.add_patch(api_box)
ax.text(5, 8.1, 'FLASK REST API (Gunicorn)', 
        ha='center', fontsize=11, fontweight='bold')

# Endpoints
endpoint_y = 7.3
endpoints = ['/health', '/classes', '/predict', '/predict_batch']
for i, endpoint in enumerate(endpoints):
    x_pos = 1.5 + i * 2
    ep_box = FancyBboxPatch((x_pos, endpoint_y), 1.5, 0.3,
                           boxstyle="round,pad=0.02",
                           facecolor='white',
                           edgecolor='gray', linewidth=1)
    ax.add_patch(ep_box)
    ax.text(x_pos + 0.75, endpoint_y + 0.15, endpoint,
           ha='center', fontsize=9)

# ML Model
model_box = FancyBboxPatch((2, 6.7), 6, 0.5,
                          boxstyle="round,pad=0.05",
                          facecolor=model_color,
                          edgecolor='black', linewidth=2)
ax.add_patch(model_box)
ax.text(5, 6.95, 'ML MODEL: MobileNetV2 | 91.78% Accuracy | 10 Categories',
       ha='center', fontsize=10, fontweight='bold')

# Arrow from cloud to local
arrow2 = FancyArrowPatch((5, 6.5), (5, 5.8),
                        arrowstyle='->', mutation_scale=30,
                        linewidth=2, color='black')
ax.add_patch(arrow2)
ax.text(5.3, 6.15, 'Results', fontsize=9)

# 3. Batch Processing Layer
batch_box = FancyBboxPatch((0.5, 4), 9, 1.6,
                          boxstyle="round,pad=0.1",
                          facecolor=batch_color,
                          edgecolor='#8E44AD', linewidth=3)
ax.add_patch(batch_box)
ax.text(5, 5.4, 'BATCH PROCESSOR (Local/Scheduled)',
       ha='center', fontsize=12, fontweight='bold', color='#6C3483')

ax.text(5, 4.9, '• Scans data/incoming/ folder\n• Sends to Cloud API\n• Organizes by category\n• Generates reports & logs',
       ha='center', fontsize=9, style='italic')

# Task Scheduler
scheduler_box = FancyBboxPatch((1, 4.1), 3, 0.4,
                              boxstyle="round,pad=0.05",
                              facecolor='#F9E79F',
                              edgecolor='black', linewidth=1)
ax.add_patch(scheduler_box)
ax.text(2.5, 4.3, 'Task Scheduler\n(Daily 2:00 AM)',
       ha='center', fontsize=8)

# 4. Data Storage Layer
storage_y = 2.5
# Incoming
incoming_box = FancyBboxPatch((0.7, storage_y), 2, 1,
                             boxstyle="round,pad=0.05",
                             facecolor=data_color,
                             edgecolor='black', linewidth=2)
ax.add_patch(incoming_box)
ax.text(1.7, storage_y + 0.7, 'DATA STORAGE',
       ha='center', fontsize=10, fontweight='bold')
ax.text(1.7, storage_y + 0.35, 'data/incoming/\ndata/processed/',
       ha='center', fontsize=8)

# Results
results_box = FancyBboxPatch((3.2, storage_y), 2.3, 1,
                            boxstyle="round,pad=0.05",
                            facecolor=data_color,
                            edgecolor='black', linewidth=2)
ax.add_patch(results_box)
ax.text(4.35, storage_y + 0.7, 'RESULTS & LOGS',
       ha='center', fontsize=10, fontweight='bold')
ax.text(4.35, storage_y + 0.35, 'batch_results/\nlogs/',
       ha='center', fontsize=8)

# Models
models_box = FancyBboxPatch((6, storage_y), 2, 1,
                           boxstyle="round,pad=0.05",
                           facecolor=data_color,
                           edgecolor='black', linewidth=2)
ax.add_patch(models_box)
ax.text(7, storage_y + 0.7, 'MODELS',
       ha='center', fontsize=10, fontweight='bold')
ax.text(7, storage_y + 0.35, 'models/\n(deployed)',
       ha='center', fontsize=8)

# GitHub
github_box = FancyBboxPatch((8.5, storage_y), 1.2, 1,
                           boxstyle="round,pad=0.05",
                           facecolor='#F5F5F5',
                           edgecolor='black', linewidth=2)
ax.add_patch(github_box)
ax.text(9.1, storage_y + 0.7, 'GITHUB',
       ha='center', fontsize=10, fontweight='bold')
ax.text(9.1, storage_y + 0.35, 'Code &\nDocs',
       ha='center', fontsize=8)

# Arrows from batch to storage
arrow3 = FancyArrowPatch((2, 4), (1.7, 3.5),
                        arrowstyle='<->', mutation_scale=20,
                        linewidth=1.5, color='gray')
ax.add_patch(arrow3)

arrow4 = FancyArrowPatch((5, 4), (4.35, 3.5),
                        arrowstyle='->', mutation_scale=20,
                        linewidth=1.5, color='gray')
ax.add_patch(arrow4)

# 5. Training Pipeline (bottom)
training_box = FancyBboxPatch((0.5, 0.5), 9, 1.3,
                             boxstyle="round,pad=0.1",
                             facecolor='#FEF9E7',
                             edgecolor='#F39C12', linewidth=3)
ax.add_patch(training_box)
ax.text(5, 1.6, 'TRAINING PIPELINE (Kaggle Notebooks with GPU)',
       ha='center', fontsize=12, fontweight='bold', color='#D68910')

# Training steps
steps = ['Dataset\n(Kaggle)', 'Data Prep\n(70/15/15)', 'Training\n(MobileNetV2)', 
         'Evaluation\n(91.78%)', 'Export\nModel', 'Deploy to\nCloud']
for i, step in enumerate(steps):
    x_pos = 1 + i * 1.4
    step_box = FancyBboxPatch((x_pos, 0.7), 1.2, 0.9,
                             boxstyle="round,pad=0.03",
                             facecolor='white',
                             edgecolor='gray', linewidth=1)
    ax.add_patch(step_box)
    ax.text(x_pos + 0.6, 1.15, step,
           ha='center', fontsize=7)
    
    if i < len(steps) - 1:
        arr = FancyArrowPatch((x_pos + 1.2, 1.15), (x_pos + 1.4, 1.15),
                             arrowstyle='->', mutation_scale=15,
                             linewidth=1, color='gray')
        ax.add_patch(arr)

# Legend
legend_elements = [
    mpatches.Patch(facecolor=cloud_color, edgecolor='#3498DB', label='Cloud Infrastructure'),
    mpatches.Patch(facecolor=api_color, edgecolor='black', label='API Layer'),
    mpatches.Patch(facecolor=batch_color, edgecolor='#8E44AD', label='Batch Processing'),
    mpatches.Patch(facecolor=data_color, edgecolor='black', label='Data Storage'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

plt.tight_layout()

# Save
os.makedirs('docs', exist_ok=True)
plt.savefig('docs/architecture_diagram.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Architecture diagram saved to: docs/architecture_diagram.png")

plt.show()