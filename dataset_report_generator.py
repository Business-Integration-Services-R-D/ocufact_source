import os
from datasets import load_from_disk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import io

# Global configuration
DATASET_PATH = "merged_ner_dataset"  # Change this to your dataset path
OUTPUT_PDF = "merged_ner_dataset_report.pdf"

def load_dataset():
    """Load the dataset from local disk"""
    try:
        dataset = load_from_disk(DATASET_PATH)
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def merge_bio_labels(label):
    """
    Merge B- and I- tags of the same entity type
    e.g., 'B-NAME' and 'I-NAME' both become 'NAME'
    """
    if label.startswith('B-') or label.startswith('I-'):
        return label[2:]  # Remove B- or I- prefix
    return label  # Keep O and other labels as is

def analyze_dataset(dataset):
    """Analyze dataset and extract statistics"""
    stats = {}
    
    # Basic dataset info
    stats['total_splits'] = list(dataset.keys())
    stats['splits_info'] = {}
    
    label_counts = {}
    total_entities = 0
    
    for split_name in dataset.keys():
        split_data = dataset[split_name]
        
        # Count samples
        stats['splits_info'][split_name] = {
            'num_samples': len(split_data),
            'labels': []
        }
        
        # Extract labels from each sample and merge BIO tags
        split_labels = []
        for sample in split_data:
            if 'ner_tags' in sample:
                labels = [split_data.features['ner_tags'].feature.int2str(tag) for tag in sample['ner_tags'] if tag != -100]
                # Merge BIO tags
                merged_labels = [merge_bio_labels(label) for label in labels]
                split_labels.extend(merged_labels)
                total_entities += len(merged_labels)
        
        stats['splits_info'][split_name]['labels'] = split_labels
        label_counts[split_name] = dict(Counter(split_labels))
    
    stats['total_entities'] = total_entities
    stats['label_counts'] = label_counts
    
    # Get unique labels across all splits
    all_labels = set()
    for split_labels in label_counts.values():
        all_labels.update(split_labels.keys())
    stats['unique_labels'] = sorted(list(all_labels))
    
    return stats

def create_visualizations(stats):
    """Create charts and graphs for the report"""
    figures = {}
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Dataset size comparison
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    splits = list(stats['splits_info'].keys())
    sizes = [stats['splits_info'][split]['num_samples'] for split in splits]
    
    bars = ax1.bar(splits, sizes, color=sns.color_palette("husl", len(splits)))
    ax1.set_title('Dataset Size by Split', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Number of Samples', fontsize=12)
    ax1.set_xlabel('Split', fontsize=12)
    
    # Add value labels on bars
    for bar, size in zip(bars, sizes):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(sizes),
                f'{size:,}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    figures['dataset_size'] = fig1
    
    # 2. Label distribution across splits (Top 20 labels)
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    
    # Create DataFrame for easier plotting
    label_data = []
    for split_name, label_count in stats['label_counts'].items():
        for label, count in label_count.items():
            label_data.append({'Split': split_name, 'Label': label, 'Count': count})
    
    df = pd.DataFrame(label_data)
    
    if not df.empty:
        # Get top 20 labels by total count
        total_counts = df.groupby('Label')['Count'].sum().sort_values(ascending=False).head(20)
        top_labels = total_counts.index.tolist()
        
        # Filter dataframe to only include top 20 labels
        df_top = df[df['Label'].isin(top_labels)]
        pivot_df = df_top.pivot(index='Label', columns='Split', values='Count').fillna(0)
        
        # Reorder by total count
        pivot_df = pivot_df.loc[top_labels]
        
        pivot_df.plot(kind='bar', ax=ax2, width=0.8)
        ax2.set_title('Label Distribution Across Splits (Top 20 Labels)', fontsize=16, fontweight='bold')
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_xlabel('Labels', fontsize=12)
        ax2.legend(title='Split')
        ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    figures['label_distribution'] = fig2
    
    # 3. Total label counts pie chart (Top 20 labels)
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    
    total_label_counts = {}
    for split_counts in stats['label_counts'].values():
        for label, count in split_counts.items():
            total_label_counts[label] = total_label_counts.get(label, 0) + count
    
    if total_label_counts:
        # Get top 20 labels and group the rest as "Others"
        sorted_labels = sorted(total_label_counts.items(), key=lambda x: x[1], reverse=True)
        top_20_labels = sorted_labels[:20]
        
        labels = [item[0] for item in top_20_labels]
        sizes = [item[1] for item in top_20_labels]
        
        # Add "Others" category if there are more than 20 labels
        if len(sorted_labels) > 20:
            others_count = sum(item[1] for item in sorted_labels[20:])
            labels.append('Others')
            sizes.append(others_count)
        
        colors_pie = sns.color_palette("husl", len(labels))
        
        wedges, texts, autotexts = ax3.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                          colors=colors_pie, startangle=90)
        ax3.set_title('Overall Label Distribution (Top 20 Labels)', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    figures['pie_chart'] = fig3
    
    # 4. Label frequency bar chart (Top 20 labels)
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    
    if total_label_counts:
        # Get top 20 labels by frequency
        sorted_items = sorted(total_label_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        labels = [item[0] for item in sorted_items]
        counts = [item[1] for item in sorted_items]
        
        bars = ax4.bar(labels, counts, color=sns.color_palette("viridis", len(labels)))
        ax4.set_title('Total Label Frequencies (Top 20 Labels)', fontsize=16, fontweight='bold')
        ax4.set_ylabel('Count', fontsize=12)
        ax4.set_xlabel('Labels', fontsize=12)
        ax4.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(counts),
                    f'{count:,}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    figures['label_frequencies'] = fig4
    
    return figures

def save_figure_to_image(fig, filename):
    """Save matplotlib figure to image"""
    fig.savefig(filename, format='png', bbox_inches='tight', dpi=300)
    plt.close(fig)

def create_text_summary(stats):
    """Create text summary of the dataset"""
    summary = []
    
    # Dataset overview
    total_samples = sum(stats['splits_info'][split]['num_samples'] for split in stats['splits_info'])
    summary.append(f"Dataset Overview:")
    summary.append(f"• Total samples: {total_samples:,}")
    summary.append(f"• Total entities: {stats['total_entities']:,}")
    summary.append(f"• Number of splits: {len(stats['splits_info'])}")
    summary.append(f"• Unique labels: {len(stats['unique_labels'])}")
    summary.append("")
    
    # Split details
    summary.append("Split Details:")
    for split_name, split_info in stats['splits_info'].items():
        summary.append(f"• {split_name}: {split_info['num_samples']:,} samples")
    summary.append("")
    
    # Label information
    summary.append("Label Information:")
    for label in stats['unique_labels']:
        total_count = sum(stats['label_counts'][split].get(label, 0) for split in stats['splits_info'].keys())
        summary.append(f"• {label}: {total_count:,} occurrences")
    
    return summary

def generate_pdf_report(stats, figures):
    """Generate the comprehensive PDF report"""
    doc = SimpleDocTemplate(OUTPUT_PDF, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=1  # Center alignment
    )
    story.append(Paragraph("Dataset Analysis Report", title_style))
    story.append(Spacer(1, 12))
    
    # Dataset path info
    story.append(Paragraph(f"Dataset Path: {DATASET_PATH}", styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Text summary
    summary = create_text_summary(stats)
    for line in summary:
        if line.strip():
            if line.startswith('•'):
                story.append(Paragraph(line, styles['Normal']))
            else:
                story.append(Paragraph(line, styles['Heading2']))
    story.append(Spacer(1, 12))
    
    # Add tables for detailed statistics
    # Split statistics table
    story.append(Paragraph("Split Statistics", styles['Heading2']))
    split_table_data = [['Split', 'Samples', 'Entities']]
    for split_name, split_info in stats['splits_info'].items():
        entity_count = sum(stats['label_counts'][split_name].values())
        split_table_data.append([split_name, f"{split_info['num_samples']:,}", f"{entity_count:,}"])
    
    split_table = Table(split_table_data)
    split_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
        ('ALIGN', (0, 1), (0, -1), 'LEFT'),
    ]))
    story.append(split_table)
    story.append(Spacer(1, 12))
    
    # Label distribution table by split
    story.append(Paragraph("Label Distribution by Split", styles['Heading2']))
    if stats['unique_labels']:
        label_table_data = [['Label'] + list(stats['splits_info'].keys())]
        for label in stats['unique_labels']:
            row = [label]
            for split_name in stats['splits_info'].keys():
                count = stats['label_counts'][split_name].get(label, 0)
                row.append(f"{count:,}")
            label_table_data.append(row)
        
        label_table = Table(label_table_data)
        label_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
            ('ALIGN', (0, 1), (0, -1), 'LEFT'),
        ]))
        story.append(label_table)
        story.append(Spacer(1, 12))
    
    # Add page break before charts
    story.append(PageBreak())
    
    # Add charts
    story.append(Paragraph("Visualizations", styles['Heading1']))
    story.append(Spacer(1, 12))
    
    # Save and embed charts
    chart_files = []
    for chart_name, fig in figures.items():
        chart_filename = f"{chart_name}.png"
        save_figure_to_image(fig, chart_filename)
        chart_files.append(chart_filename)
        
        # Add chart description
        descriptions = {
            'dataset_size': 'Dataset Size Comparison by Split',
            'label_distribution': 'Label Distribution Across Splits (Top 20 Labels)',
            'pie_chart': 'Overall Label Distribution (Top 20 Labels)',
            'label_frequencies': 'Total Label Frequencies (Top 20 Labels)'
        }
        
        story.append(Paragraph(descriptions.get(chart_name, chart_name), styles['Heading2']))
        story.append(Spacer(1, 6))
        
        # Add the image to the PDF
        try:
            img = Image(chart_filename, width=6*inch, height=4*inch)
            story.append(img)
            story.append(Spacer(1, 12))
        except Exception as e:
            print(f"Error adding chart {chart_name}: {e}")
            story.append(Paragraph(f"Error loading chart: {chart_name}", styles['Normal']))
            story.append(Spacer(1, 12))
    
    # Build PDF
    doc.build(story)
    
    # Clean up temporary chart files
    for chart_file in chart_files:
        if os.path.exists(chart_file):
            os.remove(chart_file)

def main():
    """Main function to generate the dataset report"""
    print("Loading dataset...")
    dataset = load_dataset()
    
    if dataset is None:
        print("Failed to load dataset. Please check the DATASET_PATH.")
        return
    
    print("Analyzing dataset...")
    stats = analyze_dataset(dataset)
    
    print("Creating visualizations...")
    figures = create_visualizations(stats)
    
    print("Generating PDF report...")
    generate_pdf_report(stats, figures)
    
    print(f"Report generated successfully: {OUTPUT_PDF}")

if __name__ == "__main__":
    main()
