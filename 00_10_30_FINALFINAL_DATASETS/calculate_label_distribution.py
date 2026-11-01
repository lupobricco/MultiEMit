import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_ironia_sarcasmo_stats(df):
    """
    Calculates detailed statistics for Ironia and Sarcasmo combinations.
    
    Args:
        df (DataFrame): The dataset dataframe
        
    Returns:
        dict: Dictionary containing various counts and percentages
    """
    total_examples = len(df)
    
    # Check if columns exist
    if 'Ironia' not in df.columns or 'Sarcasmo' not in df.columns:
        print("Warning: 'Ironia' or 'Sarcasmo' columns not found in dataset")
        return {}
    
    # Calculate combinations
    only_ironia = len(df[(df['Ironia'] == 1) & (df['Sarcasmo'] == 0)])
    only_sarcasmo = len(df[(df['Ironia'] == 0) & (df['Sarcasmo'] == 1)])
    both = len(df[(df['Ironia'] == 1) & (df['Sarcasmo'] == 1)])
    neither = len(df[(df['Ironia'] == 0) & (df['Sarcasmo'] == 0)])
    
    # Total counts (including overlaps)
    total_ironia = df['Ironia'].sum()
    total_sarcasmo = df['Sarcasmo'].sum()
    
    # Calculate percentages
    stats = {
        'only_ironia': only_ironia,
        'only_ironia_pct': (only_ironia / total_examples) * 100,
        'only_sarcasmo': only_sarcasmo,
        'only_sarcasmo_pct': (only_sarcasmo / total_examples) * 100,
        'both': both,
        'both_pct': (both / total_examples) * 100,
        'neither': neither,
        'neither_pct': (neither / total_examples) * 100,
        'total_ironia': total_ironia,
        'total_ironia_pct': (total_ironia / total_examples) * 100,
        'total_sarcasmo': total_sarcasmo,
        'total_sarcasmo_pct': (total_sarcasmo / total_examples) * 100
    }
    
    return stats

def calculate_label_distribution(csv_file):
    """
    Calculates the distribution of labels in the dataset.
    Counts how many examples have each label (value = 1).
    
    Args:
        csv_file (str): Path to the CSV file
    """
    
    # Read the CSV file
    print(f"Reading {csv_file}...")
    df = pd.read_csv(csv_file)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Total examples: {len(df)}")
    
    # Get label columns (exclude 'uri' and 'text' columns)
    label_columns = [col for col in df.columns if col not in ['uri', 'text', 'img', 'img_path']]
    
    print(f"\nLabel columns found: {len(label_columns)}")
    print(f"Labels: {label_columns}")
    
    # Calculate distribution (count of 1s for each label)
    label_counts = {}
    label_percentages = {}
    
    for label in label_columns:
        count = df[label].sum()  # Sum of 1s
        percentage = (count / len(df)) * 100
        label_counts[label] = count
        label_percentages[label] = percentage
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'Label': label_columns,
        'Count': [label_counts[label] for label in label_columns],
        'Percentage': [label_percentages[label] for label in label_columns]
    })
    
    # Sort by count (descending)
    results_df = results_df.sort_values('Count', ascending=False)
    
    # Display results
    print("\n" + "="*60)
    print("LABEL DISTRIBUTION RESULTS")
    print("="*60)
    print(f"{'Label':<20} {'Count':<8} {'Percentage':<12}")
    print("-" * 45)
    
    for _, row in results_df.iterrows():
        print(f"{row['Label']:<20} {row['Count']:<8} {row['Percentage']:<12.2f}%")
    
    # Calculate some statistics
    total_labels = results_df['Count'].sum()
    avg_labels_per_example = total_labels / len(df)
    
    # Calculate Ironia and Sarcasmo combinations
    ironia_sarcasmo_stats = calculate_ironia_sarcasmo_stats(df)
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total label instances: {total_labels}")
    print(f"Average labels per example: {avg_labels_per_example:.2f}")
    print(f"Most common label: {results_df.iloc[0]['Label']} ({results_df.iloc[0]['Count']} instances)")
    print(f"Least common label: {results_df.iloc[-1]['Label']} ({results_df.iloc[-1]['Count']} instances)")
    
    print("\n" + "="*60)
    print("IRONIA & SARCASMO ANALYSIS")
    print("="*60)
    print(f"Only Ironia (no Sarcasmo): {ironia_sarcasmo_stats['only_ironia']} examples ({ironia_sarcasmo_stats['only_ironia_pct']:.2f}%)")
    print(f"Only Sarcasmo (no Ironia): {ironia_sarcasmo_stats['only_sarcasmo']} examples ({ironia_sarcasmo_stats['only_sarcasmo_pct']:.2f}%)")
    print(f"Both Ironia and Sarcasmo: {ironia_sarcasmo_stats['both']} examples ({ironia_sarcasmo_stats['both_pct']:.2f}%)")
    print(f"Neither Ironia nor Sarcasmo: {ironia_sarcasmo_stats['neither']} examples ({ironia_sarcasmo_stats['neither_pct']:.2f}%)")
    print(f"Total with Ironia (including both): {ironia_sarcasmo_stats['total_ironia']} examples ({ironia_sarcasmo_stats['total_ironia_pct']:.2f}%)")
    print(f"Total with Sarcasmo (including both): {ironia_sarcasmo_stats['total_sarcasmo']} examples ({ironia_sarcasmo_stats['total_sarcasmo_pct']:.2f}%)")
    
    # Save results to CSV
    output_file = csv_file.replace('.csv', '_label_distribution.csv')
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    # Save Ironia/Sarcasmo analysis to separate CSV
    if ironia_sarcasmo_stats:
        ironia_sarcasmo_df = pd.DataFrame([
            {'Category': 'Only Ironia', 'Count': ironia_sarcasmo_stats['only_ironia'], 'Percentage': ironia_sarcasmo_stats['only_ironia_pct']},
            {'Category': 'Only Sarcasmo', 'Count': ironia_sarcasmo_stats['only_sarcasmo'], 'Percentage': ironia_sarcasmo_stats['only_sarcasmo_pct']},
            {'Category': 'Both Ironia and Sarcasmo', 'Count': ironia_sarcasmo_stats['both'], 'Percentage': ironia_sarcasmo_stats['both_pct']},
            {'Category': 'Neither Ironia nor Sarcasmo', 'Count': ironia_sarcasmo_stats['neither'], 'Percentage': ironia_sarcasmo_stats['neither_pct']},
            {'Category': 'Total with Ironia', 'Count': ironia_sarcasmo_stats['total_ironia'], 'Percentage': ironia_sarcasmo_stats['total_ironia_pct']},
            {'Category': 'Total with Sarcasmo', 'Count': ironia_sarcasmo_stats['total_sarcasmo'], 'Percentage': ironia_sarcasmo_stats['total_sarcasmo_pct']}
        ])
        ironia_sarcasmo_file = csv_file.replace('.csv', '_ironia_sarcasmo_analysis.csv')
        ironia_sarcasmo_df.to_csv(ironia_sarcasmo_file, index=False)
        print(f"Ironia/Sarcasmo analysis saved to: {ironia_sarcasmo_file}")
    
    # Create and save visualization
    create_visualization(results_df, csv_file)
    
    return results_df

def create_visualization(results_df, original_file):
    """
    Creates a bar plot of the label distribution.
    """
    plt.figure(figsize=(15, 8))
    
    # Create bar plot
    bars = plt.bar(range(len(results_df)), results_df['Count'], 
                   color='skyblue', edgecolor='navy', alpha=0.7)
    
    # Customize the plot
    plt.title('Label Distribution in Dataset', fontsize=16, fontweight='bold')
    plt.xlabel('Labels', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(range(len(results_df)), results_df['Label'], rotation=45, ha='right')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = original_file.replace('.csv', '_label_distribution.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {plot_file}")
    
    plt.show()

if __name__ == "__main__":
    # Input file path
    input_file = "00_10_30_FINALFINAL_DATASETS/textual_only_cleaned.csv"
    
    try:
        # Calculate distribution
        results = calculate_label_distribution(input_file)
        
    except FileNotFoundError:
        print(f"Error: File {input_file} not found!")
        print("Please make sure you're running this script from the correct directory.")
    except Exception as e:
        print(f"Error processing file: {e}")