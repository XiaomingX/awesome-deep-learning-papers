import re
import requests
from html.parser import HTMLParser
import codecs

# Constants
SEARCH_ENGINE_URL = "https://www.semanticscholar.org/search?q="
POST_FIX = "&sort=relevance&ae=false"

class AuthorParser(HTMLParser):
    """Parser to extract author names from the HTML content of Semantic Scholar search results."""

    def __init__(self):
        super().__init__()
        self.tail_string = ""  # Keeps track of the tag path to identify authors
        self.m_Stop = False  # Flag to stop after processing the first article
        self.m_authors = []  # List to store author names

    def handle_starttag(self, tag, attrs):
        """Handles the start of a tag."""
        if self.m_Stop:
            return

        if tag == 'article':
            self.tail_string += tag
            return

        if self.tail_string != "":
            self.tail_string = self.tail_string + "." + tag

    def handle_endtag(self, tag):
        """Handles the end of a tag."""
        if self.m_Stop:
            return

        if self.tail_string == "article":
            # Stop after processing the first article
            self.m_Stop = True

        if self.tail_string != "":
            tags = self.tail_string.split('.')
            tags.reverse()
            for t in tags:
                if t == tag:
                    tags.remove(t)
                    break
            self.tail_string = ""
            tags.reverse()
            for i, t in enumerate(tags):
                self.tail_string = self.tail_string + "." + t if i > 0 else t

    def handle_data(self, data):
        """Handles the data inside a tag."""
        if self.m_Stop:
            return

        if self.tail_string == "article.header.ul.li.span.span.a.span.span":
            # Extract authors' names from the relevant section
            self.m_authors.append(data)

    def get_authors(self):
        """Returns the list of authors."""
        return self.m_authors

    def clean(self):
        """Resets the parser state for the next article."""
        self.m_authors = []
        self.tail_string = ""
        self.m_Stop = False


def get_paper_names(readme_file):
    """Extracts paper names from the provided README file."""
    paper_list = []
    with codecs.open(readme_file, encoding='utf-8', mode='r', buffering=1, errors='strict') as f:
        lines = f.read().split('\n')
        heading, section_path = '', ''
        
        for line in lines:
            if '###' in line:
                heading = line.strip().split('###')[1].replace('/', '|')

            if '[[pdf]]' in line:
                # Regular expression to extract paper title and URL
                result = re.search(r'\*\*(.*?)\*\*.*?\[\[pdf\]\]\((.*?)\)', line)
                if result:
                    paper, url = result.groups()
                    paper_list.append(paper)

    return paper_list


def main():
    """Main function to tie everything together."""
    # Get paper names from README
    all_papers = get_paper_names("README.md")

    # Initialize the author parser
    author_parser = AuthorParser()

    # Dictionary to store authors and their papers
    author_dict = {}

    for index, paper in enumerate(all_papers):
        # Replace spaces with %20 for URL compatibility
        paper_query = paper.replace(" ", "%20")
        
        # Make a request to the search engine
        search_url = SEARCH_ENGINE_URL + paper_query + POST_FIX
        search_result = requests.get(search_url)
        
        # Feed the result to the parser
        author_parser.feed(search_result.text)
        
        # Get authors from the parsed HTML
        authors = author_parser.get_authors()

        # Update the author dictionary with paper information
        for weight, author in enumerate(authors):
            if author not in author_dict:
                author_dict[author] = []
            author_dict[author].append((weight + 1, paper))
        
        # Reset the parser state for the next paper
        author_parser.clean()

        # Print progress
        print(f"Processed {index + 1} | {paper}")

    # Write the results to a CSV file
    with open("author.csv", 'w') as fcsv:
        for author, papers in author_dict.items():
            score = sum(1.0 / weight for weight, _ in papers)
            print(f"{author} score: {score:.2f}")
            fcsv.write(f"{author},{score:.2f}\n")


# Run the main function
if __name__ == "__main__":
    main()
