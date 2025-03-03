#!/usr/bin/env python3
"""
Generate an enhanced HTML report from the test results.
"""

import os
import sys
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import base64
from io import BytesIO

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate HTML report from test results")
    parser.add_argument("--input", default="direct_test_results",
                        help="Input directory containing test results")
    parser.add_argument("--output", default=None,
                        help="Output HTML file (default: input_dir/enhanced_report.html)")
    parser.add_argument("--title", default="Synthetic Conversation Evaluation Report",
                        help="Title for the report")
    return parser.parse_args()

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string for embedding in HTML."""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_str

def generate_visualizations(df):
    """Generate visualizations for the report."""
    visuals = {}
    
    # Skill scores by skill level
    if len(df['skill_level'].unique()) > 1:
        fig = plt.figure(figsize=(10, 6))
        skill_levels = sorted(df['skill_level'].unique())
        
        # Set bar width
        bar_width = 0.25
        
        # Set position of bars on x axis
        r1 = range(len(skill_levels))
        r2 = [x + bar_width for x in r1]
        r3 = [x + bar_width for x in r2]
        
        # Create bars
        critical_thinking = [df[df['skill_level'] == level]['critical_thinking'].mean() for level in skill_levels]
        communication = [df[df['skill_level'] == level]['communication'].mean() for level in skill_levels]
        emotional_intelligence = [df[df['skill_level'] == level]['emotional_intelligence'].mean() for level in skill_levels]
        
        plt.bar(r1, critical_thinking, width=bar_width, label='Critical Thinking', color='#3498db')
        plt.bar(r2, communication, width=bar_width, label='Communication', color='#2ecc71')
        plt.bar(r3, emotional_intelligence, width=bar_width, label='Emotional Intelligence', color='#e74c3c')
        
        # Add labels and legend
        plt.xlabel('Skill Level')
        plt.ylabel('Score')
        plt.title('Skill Scores by Skill Level')
        plt.xticks([r + bar_width for r in range(len(skill_levels))], [level.capitalize() for level in skill_levels])
        plt.legend()
        
        visuals['skill_scores'] = fig_to_base64(fig)
        plt.close(fig)
    
    # Total scores by skill level
    if len(df['skill_level'].unique()) > 1:
        fig = plt.figure(figsize=(10, 6))
        skill_levels = sorted(df['skill_level'].unique())
        
        # Set bar width
        bar_width = 0.4
        
        # Set position of bars on x axis
        r1 = range(len(skill_levels))
        r2 = [x + bar_width for x in r1]
        
        # Create bars
        total_scores = [df[df['skill_level'] == level]['total_score'].mean() for level in skill_levels]
        percentile_scores = [df[df['skill_level'] == level]['percentile_score'].mean() for level in skill_levels]
        
        plt.bar(r1, total_scores, width=bar_width, label='Total Score', color='#9b59b6')
        plt.bar(r2, percentile_scores, width=bar_width, label='Percentile Score', color='#f39c12')
        
        # Add labels and legend
        plt.xlabel('Skill Level')
        plt.ylabel('Score')
        plt.title('Total Scores by Skill Level')
        plt.xticks([r + bar_width/2 for r in range(len(skill_levels))], [level.capitalize() for level in skill_levels])
        plt.legend()
        
        visuals['total_scores'] = fig_to_base64(fig)
        plt.close(fig)
    
    # Badge distribution pie chart
    if len(df) > 1:
        fig = plt.figure(figsize=(8, 8))
        badge_counts = df['badge_level'].value_counts()
        colors = {'Bronze': '#CD7F32', 'Silver': '#C0C0C0', 'Gold': '#FFD700'}
        plt_colors = [colors.get(badge, '#CCCCCC') for badge in badge_counts.index]
        plt.pie(badge_counts, labels=badge_counts.index, autopct='%1.1f%%', colors=plt_colors)
        plt.title('Badge Level Distribution')
        
        visuals['badge_distribution'] = fig_to_base64(fig)
        plt.close(fig)
    
    # Radar chart for skill areas
    if len(df) > 0:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, polar=True)
        
        # Skills to include
        skills = ['critical_thinking', 'communication', 'emotional_intelligence']
        
        # Number of skills
        N = len(skills)
        
        # Compute mean score for each skill
        mean_scores = [df[skill].mean() for skill in skills]
        
        # Compute max possible score for scaling
        max_score = max(df[skills].max())
        
        # Compute angles for each skill
        angles = [n / float(N) * 2 * 3.14159 for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Add the scores to the chart
        mean_scores += mean_scores[:1]  # Close the loop
        ax.plot(angles, mean_scores, linewidth=1, linestyle='solid', color='#3498db')
        ax.fill(angles, mean_scores, alpha=0.1, color='#3498db')
        
        # Set labels
        plt.xticks(angles[:-1], [skill.replace('_', ' ').title() for skill in skills])
        
        # Set y-ticks
        ax.set_rlabel_position(0)
        plt.yticks([max_score/4, max_score/2, 3*max_score/4, max_score], 
                  [f"{max_score/4:.1f}", f"{max_score/2:.1f}", f"{3*max_score/4:.1f}", f"{max_score:.1f}"], 
                  color="grey", size=7)
        plt.ylim(0, max_score)
        
        plt.title('Average Skill Scores')
        
        visuals['radar_chart'] = fig_to_base64(fig)
        plt.close(fig)
    
    return visuals

def generate_html_report(args):
    """Generate an enhanced HTML report from the test results."""
    input_dir = args.input
    output_file = args.output or os.path.join(input_dir, "enhanced_report.html")
    
    # Check if the input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' not found.")
        return False
    
    # Find all conversation files
    conversation_files = [f for f in os.listdir(input_dir) 
                         if f.startswith("conversation_") and f.endswith(".json")]
    
    if not conversation_files:
        print(f"Error: No conversation files found in '{input_dir}'.")
        return False
    
    print(f"Generating enhanced HTML report from {len(conversation_files)} conversation files...")
    
    # Load summary data if available
    summary_file = os.path.join(input_dir, "test_results_summary.csv")
    if os.path.exists(summary_file):
        df = pd.read_csv(summary_file)
    else:
        # Create summary data from conversation files
        summary_data = []
        for file_name in conversation_files:
            file_path = os.path.join(input_dir, file_name)
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            evaluation = data.get("evaluation", {})
            
            summary_data.append({
                "conversation_id": data.get("conversation_id", "Unknown"),
                "skill_level": data.get("skill_level", "Unknown"),
                "gradient": data.get("gradient", 0.0),
                "total_score": evaluation.get("total_score", 0),
                "percentile_score": evaluation.get("percentile_score", 0),
                "badge_level": evaluation.get("badge_level", "Bronze"),
                "critical_thinking": evaluation.get("skill_scores", {}).get("critical_thinking", 0),
                "communication": evaluation.get("skill_scores", {}).get("communication", 0),
                "emotional_intelligence": evaluation.get("skill_scores", {}).get("emotional_intelligence", 0)
            })
        
        df = pd.DataFrame(summary_data)
    
    # Generate visualizations
    visuals = generate_visualizations(df)
    
    # Generate HTML report
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{args.title}</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap" rel="stylesheet">
        <style>
            body {{
                font-family: 'Roboto', sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 0;
                color: #333;
                background-color: #f8f9fa;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            .header {{
                background-color: #343a40;
                color: white;
                padding: 2rem;
                border-radius: 5px;
                margin-bottom: 30px;
                text-align: center;
            }}
            .section {{
                background-color: #fff;
                border-radius: 5px;
                padding: 20px;
                margin-bottom: 30px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .section-title {{
                font-size: 1.5rem;
                margin-bottom: 20px;
                padding-bottom: 10px;
                border-bottom: 1px solid #eee;
                color: #343a40;
            }}
            .conversation {{
                background-color: #fff;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 20px;
                margin-bottom: 30px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .conversation-header {{
                background-color: #f8f9fa;
                padding: 15px;
                margin-bottom: 15px;
                border-radius: 5px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            .conversation-title {{
                font-size: 1.2rem;
                font-weight: 500;
                margin: 0;
            }}
            .badge {{
                display: inline-block;
                padding: 5px 10px;
                border-radius: 5px;
                color: white;
                font-weight: 500;
                text-align: center;
                min-width: 80px;
            }}
            .badge-Bronze {{
                background-color: #CD7F32;
            }}
            .badge-Silver {{
                background-color: #C0C0C0;
            }}
            .badge-Gold {{
                background-color: #FFD700;
                color: #333;
            }}
            .skill-badge {{
                font-size: 0.8rem;
                padding: 3px 8px;
                border-radius: 3px;
                margin-left: 5px;
            }}
            .exchange {{
                margin-bottom: 20px;
                padding-bottom: 20px;
                border-bottom: 1px solid #eee;
            }}
            .exchange:last-child {{
                border-bottom: none;
            }}
            .stage-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 10px;
            }}
            .stage-label {{
                font-weight: 500;
                color: #6c757d;
            }}
            .score-pill {{
                background-color: #e9ecef;
                padding: 3px 10px;
                border-radius: 15px;
                font-size: 0.8rem;
                font-weight: 500;
            }}
            .score-0, .score-1 {{
                background-color: #f8d7da;
                color: #721c24;
            }}
            .score-2 {{
                background-color: #d1ecf1;
                color: #0c5460;
            }}
            .score-3 {{
                background-color: #d4edda;
                color: #155724;
            }}
            .message {{
                padding: 12px;
                border-radius: 5px;
                margin-bottom: 10px;
                position: relative;
            }}
            .ai-message {{
                background-color: #f1f0f0;
                margin-right: 20px;
            }}
            .user-message {{
                background-color: #e3f2fd;
                margin-left: 20px;
            }}
            .speaker-label {{
                font-weight: 500;
                margin-bottom: 5px;
                color: #495057;
            }}
            .feedback {{
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                margin-top: 10px;
                font-size: 0.9rem;
            }}
            .feedback-item {{
                margin-bottom: 10px;
            }}
            .feedback-label {{
                font-weight: 500;
                color: #6c757d;
            }}
            .improvement {{
                color: #0c5460;
                background-color: #d1ecf1;
                padding: 10px;
                border-radius: 5px;
                margin-top: 5px;
            }}
            .skill-scores {{
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
                margin-top: 20px;
            }}
            .skill-score {{
                flex: 1;
                min-width: 200px;
                padding: 15px;
                background-color: #f8f9fa;
                border-radius: 5px;
            }}
            .skill-name {{
                font-weight: 500;
                margin-bottom: 10px;
                color: #343a40;
            }}
            .progress-bar {{
                height: 10px;
                background-color: #e9ecef;
                border-radius: 5px;
                margin-top: 5px;
                overflow: hidden;
            }}
            .progress {{
                height: 100%;
                border-radius: 5px;
            }}
            .progress-critical-thinking {{
                background-color: #3498db;
            }}
            .progress-communication {{
                background-color: #2ecc71;
            }}
            .progress-emotional-intelligence {{
                background-color: #e74c3c;
            }}
            .summary-table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
            }}
            .summary-table th, .summary-table td {{
                padding: 12px 15px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            .summary-table th {{
                background-color: #f8f9fa;
                font-weight: 500;
                color: #495057;
            }}
            .summary-table tr:hover {{
                background-color: #f8f9fa;
            }}
            .summary-table .badge {{
                font-size: 0.8rem;
                padding: 3px 8px;
            }}
            .visualization {{
                text-align: center;
                margin: 30px 0;
            }}
            .visualization img {{
                max-width: 100%;
                height: auto;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .visualization-caption {{
                margin-top: 10px;
                font-style: italic;
                color: #6c757d;
            }}
            .grid-container {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
                grid-gap: 30px;
                margin-top: 30px;
            }}
            .toggle-button {{
                background-color: #007bff;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 5px;
                cursor: pointer;
                font-size: 0.9rem;
            }}
            .toggle-button:hover {{
                background-color: #0069d9;
            }}
            .toggle-content {{
                display: none;
            }}
            .filter-controls {{
                margin-bottom: 20px;
                display: flex;
                gap: 15px;
                flex-wrap: wrap;
            }}
            .filter-button {{
                background-color: #e9ecef;
                border: none;
                padding: 8px 15px;
                border-radius: 5px;
                cursor: pointer;
                font-size: 0.9rem;
            }}
            .filter-button.active {{
                background-color: #007bff;
                color: white;
            }}
            .footer {{
                text-align: center;
                padding: 20px;
                margin-top: 30px;
                color: #6c757d;
                font-size: 0.9rem;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>{args.title}</h1>
                <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
    """
    
    # Add summary section
    html += """
            <div class="section">
                <h2 class="section-title">Summary</h2>
                <p>This report contains evaluations of synthetic networking conversations at different skill levels.</p>
    """
    
    # Add visualizations if available
    if visuals:
        html += """
                <div class="grid-container">
        """
        
        if 'skill_scores' in visuals:
            html += f"""
                    <div class="visualization">
                        <h3>Skill Scores by Skill Level</h3>
                        <img src="data:image/png;base64,{visuals['skill_scores']}" alt="Skill Scores by Skill Level">
                        <p class="visualization-caption">Average scores for each skill area across different skill levels</p>
                    </div>
            """
        
        if 'total_scores' in visuals:
            html += f"""
                    <div class="visualization">
                        <h3>Total Scores by Skill Level</h3>
                        <img src="data:image/png;base64,{visuals['total_scores']}" alt="Total Scores by Skill Level">
                        <p class="visualization-caption">Average total and percentile scores across different skill levels</p>
                    </div>
            """
        
        html += """
                </div>
        """
        
        if 'badge_distribution' in visuals:
            html += f"""
                <div class="visualization">
                    <h3>Badge Level Distribution</h3>
                    <img src="data:image/png;base64,{visuals['badge_distribution']}" alt="Badge Level Distribution">
                    <p class="visualization-caption">Distribution of badge levels across all conversations</p>
                </div>
            """
        
        if 'radar_chart' in visuals:
            html += f"""
                <div class="visualization">
                    <h3>Average Skill Scores</h3>
                    <img src="data:image/png;base64,{visuals['radar_chart']}" alt="Average Skill Scores">
                    <p class="visualization-caption">Average scores for each skill area across all conversations</p>
                </div>
            """
    
    # Add summary table
    html += """
                <table class="summary-table">
                    <thead>
                        <tr>
                            <th>Conversation</th>
                            <th>Skill Level</th>
                            <th>Total Score</th>
                            <th>Badge Level</th>
                            <th>Critical Thinking</th>
                            <th>Communication</th>
                            <th>Emotional Intelligence</th>
                        </tr>
                    </thead>
                    <tbody>
    """
    
    for _, row in df.iterrows():
        html += f"""
                        <tr class="conversation-row" data-skill-level="{row['skill_level']}">
                            <td>{row['conversation_id']}</td>
                            <td>{row['skill_level'].capitalize()}</td>
                            <td>{row['total_score']}</td>
                            <td><span class="badge badge-{row['badge_level']}">{row['badge_level']}</span></td>
                            <td>{row['critical_thinking']}</td>
                            <td>{row['communication']}</td>
                            <td>{row['emotional_intelligence']}</td>
                        </tr>
        """
    
    html += """
                    </tbody>
                </table>
            </div>
    """
    
    # Add conversation details section
    html += """
            <div class="section">
                <h2 class="section-title">Detailed Conversations</h2>
                
                <div class="filter-controls">
                    <button class="filter-button active" data-filter="all">All</button>
    """
    
    for level in sorted(df['skill_level'].unique()):
        html += f"""
                    <button class="filter-button" data-filter="{level}">{level.capitalize()}</button>
        """
    
    html += """
                </div>
    """
    
    # Process each conversation file
    for file_name in sorted(conversation_files):
        file_path = os.path.join(input_dir, file_name)
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        conversation = data.get("conversation", [])
        eval_data = data.get("evaluation", {})
        skill_scores = eval_data.get("skill_scores", {})
        badge_level = eval_data.get("badge_level", "Bronze")
        skill_level = data.get("skill_level", "unknown")
        total_score = eval_data.get("total_score", 0)
        percentile_score = eval_data.get("percentile_score", 0)
        
        html += f"""
                <div class="conversation conversation-container" data-skill-level="{skill_level}">
                    <div class="conversation-header">
                        <div>
                            <h3 class="conversation-title">{data.get("conversation_id", "Unknown")}</h3>
                            <p><strong>Skill Level:</strong> {skill_level.capitalize()}</p>
                            <p><strong>Persona:</strong> {data.get("persona", "Unknown")}</p>
                        </div>
                        <div>
                            <p><strong>Total Score:</strong> {total_score}</p>
                            <p><strong>Percentile Score:</strong> {percentile_score}%</p>
                            <p><strong>Badge Level:</strong> <span class="badge badge-{badge_level}">{badge_level}</span></p>
                        </div>
                    </div>
                
                    <button class="toggle-button" onclick="toggleContent(this)">Show Conversation</button>
                    <div class="toggle-content">
        """
        
        # Add conversation exchanges
        for exchange in conversation:
            stage = exchange.get("stage", "unknown")
            ai_prompt = exchange.get("ai_prompt", "")
            user_response = exchange.get("user_response", "")
            
            # Get stage evaluation if available
            stage_eval = eval_data.get("stages", {}).get(stage, {})
            stage_score = stage_eval.get("score", 0)
            stage_feedback = stage_eval.get("feedback", "")
            stage_improvement = stage_eval.get("improvement", "")
            
            html += f"""
                        <div class="exchange">
                            <div class="stage-header">
                                <div class="stage-label">{stage.upper()}</div>
                                <div class="score-pill score-{stage_score}">Score: {stage_score}/3</div>
                            </div>
                            
                            <div class="message ai-message">
                                <div class="speaker-label">AI:</div>
                                {ai_prompt}
                            </div>
                            
                            <div class="message user-message">
                                <div class="speaker-label">User:</div>
                                {user_response}
                            </div>
                            
                            <div class="feedback">
                                <div class="feedback-item">
                                    <div class="feedback-label">Feedback:</div>
                                    {stage_feedback}
                                </div>
                                
                                <div class="feedback-item">
                                    <div class="feedback-label">Improvement:</div>
                                    <div class="improvement">{stage_improvement}</div>
                                </div>
                            </div>
                        </div>
            """
        
        # Add skill scores
        html += """
                        <div class="skill-scores">
        """
        
        for skill, score in skill_scores.items():
            skill_badge = eval_data.get("badges", {}).get(skill, "Bronze")
            max_score = 5  # Assuming max score is 5
            percentage = (score / max_score) * 100
            
            html += f"""
                            <div class="skill-score">
                                <div class="skill-name">{skill.replace('_', ' ').title()} <span class="badge skill-badge badge-{skill_badge}">{skill_badge}</span></div>
                                <p>{score}/{max_score}</p>
                                <div class="progress-bar">
                                    <div class="progress progress-{skill.replace('_', '-')}" style="width: {percentage}%;"></div>
                                </div>
                            </div>
            """
        
        html += """
                        </div>
                    </div>
                </div>
        """
    
    # Add JavaScript for interactive elements
    html += """
            </div>
            
            <div class="footer">
                <p>Generated by Synthetic Conversation Evaluation System</p>
            </div>
        </div>
        
        <script>
            function toggleContent(button) {
                const content = button.nextElementSibling;
                if (content.style.display === "block") {
                    content.style.display = "none";
                    button.textContent = "Show Conversation";
                } else {
                    content.style.display = "block";
                    button.textContent = "Hide Conversation";
                }
            }
            
            // Filter functionality
            document.addEventListener('DOMContentLoaded', function() {
                const filterButtons = document.querySelectorAll('.filter-button');
                
                filterButtons.forEach(button => {
                    button.addEventListener('click', function() {
                        // Remove active class from all buttons
                        filterButtons.forEach(btn => btn.classList.remove('active'));
                        
                        // Add active class to clicked button
                        this.classList.add('active');
                        
                        const filter = this.getAttribute('data-filter');
                        const containers = document.querySelectorAll('.conversation-container');
                        
                        containers.forEach(container => {
                            if (filter === 'all' || container.getAttribute('data-skill-level') === filter) {
                                container.style.display = 'block';
                            } else {
                                container.style.display = 'none';
                            }
                        });
                    });
                });
            });
        </script>
    </body>
    </html>
    """
    
    # Write the HTML to a file
    try:
        with open(output_file, 'w') as f:
            f.write(html)
        
        print(f"Enhanced HTML report generated: {output_file}")
        return True
    except Exception as e:
        print(f"Error writing HTML report: {e}")
        return False

if __name__ == "__main__":
    args = parse_args()
    success = generate_html_report(args)
    if success:
        print(f"HTML report generated successfully at {args.output or os.path.join(args.input, 'enhanced_report.html')}")
        # Try to open the report in a browser
        try:
            import webbrowser
            webbrowser.open(f"file://{os.path.abspath(args.output or os.path.join(args.input, 'enhanced_report.html'))}")
        except:
            pass
    else:
        print("Failed to generate HTML report.")
        sys.exit(1) 