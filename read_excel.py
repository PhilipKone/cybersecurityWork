import pandas as pd

# Read the Excel file starting from row 2 (index 1)
df = pd.read_excel('articles.xlsx', header=1)

# Remove unnamed columns and any completely empty rows
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.dropna(how='all')

# Remove rows where Title is 'Title' or empty
df = df[df['Title'].notna() & (df['Title'] != 'Title')]

# Create markdown content
markdown_content = "# Research Articles Database\n\n"

# Create comprehensive table headers
markdown_content += "| Title | Authors | Journal | Publisher | Year | Volume | Issue | Problem Solved | Methodology | Data & Availability | Performance Metrics | Future Work | Limitations | Critique |\n"
markdown_content += "|--------|---------|----------|------------|------|---------|--------|----------------|------------|-------------------|-------------------|-------------|------------|----------|"

# Process each article
for index, row in df.iterrows():
    # Clean and format all fields
    title = str(row['Title']).replace('|', '-').replace('\n', ' ') if pd.notna(row['Title']) else 'N/A'
    authors = str(row['Authors']).replace('|', '-').replace('\n', ' ') if pd.notna(row['Authors']) else 'N/A'
    journal = str(row['Journal Name']).replace('|', '-').replace('\n', ' ') if pd.notna(row['Journal Name']) else 'N/A'
    publisher = str(row['Publishing House']).replace('|', '-').replace('\n', ' ') if pd.notna(row['Publishing House']) else 'N/A'
    year = str(row['Year']).replace('|', '-').replace('\n', ' ') if pd.notna(row['Year']) else 'N/A'
    volume = str(row['Volume']).replace('|', '-').replace('\n', ' ') if pd.notna(row['Volume']) else 'N/A'
    issue = str(row['Issue']).replace('|', '-').replace('\n', ' ') if pd.notna(row['Issue']) else 'N/A'
    problem = str(row['Problem Solved']).replace('|', '-').replace('\n', ' ') if pd.notna(row['Problem Solved']) else 'N/A'
    method = str(row['Method Used to Solve the Problem']).replace('|', '-').replace('\n', ' ') if pd.notna(row['Method Used to Solve the Problem']) else 'N/A'
    data = str(row['Data & Its Availability']).replace('|', '-').replace('\n', ' ') if pd.notna(row['Data & Its Availability']) else 'N/A'
    metrics = str(row['Performance Metrics & Values']).replace('|', '-').replace('\n', ' ') if pd.notna(row['Performance Metrics & Values']) else 'N/A'
    future = str(row['Future Works (if any)']).replace('|', '-').replace('\n', ' ') if pd.notna(row['Future Works (if any)']) else 'N/A'
    limitations = str(row['Limitation (if any)']).replace('|', '-').replace('\n', ' ') if pd.notna(row['Limitation (if any)']) else 'N/A'
    critique = str(row['Critique (if any)']).replace('|', '-').replace('\n', ' ') if pd.notna(row['Critique (if any)']) else 'N/A'
    
    # Add row to table
    markdown_content += f"\n| {title} | {authors} | {journal} | {publisher} | {year} | {volume} | {issue} | {problem} | {method} | {data} | {metrics} | {future} | {limitations} | {critique} |"

# Write to markdown file
with open('research_articles_comprehensive.md', 'w', encoding='utf-8') as f:
    f.write(markdown_content)

print("Successfully converted Excel data to research_articles_comprehensive.md")
