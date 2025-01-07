# Used Fact-Checking Articles and Links
## Fact-Checking Articles
The file `fc_articles.csv` contains all crawled fact-checking article URLs together with extracted key information:
| Column | Description |
| ---- | ---- |
|**fc_id** | Unique ID for the fact-checking article. |
| **url** | URL of the fact-checking article. |
| **claim_short** | The short version of the claim as stated in the fact-checking article.|
| **claim** | The complete claim as stated in the fact-checking article.|
| **claim_source** | A link to the source with the original claim (in the wild) |
| **verdict_img** | The link to the verdict image (to get the overall verdict) |
| **date** | Date of the claim as stated in the fact-checking article. |
| **categories**| A list of verdict labels such as "misleading", "inaccurate", etc. They refer to the bold highlighted words in the verdict summary.

## Links from the fact-checking articles
The file `fc_links.csv` contains *all* the links from the fact-checking articles (before filtering):
| Column | Description |
| --- | --- |
| **fc_id** | Unique ID for the fact-checking article. |
| **fc_url** | URL of the fact-checking article. |
| **link_text** | Text of the link within the fact-checking article. |
| **link_url** | Link (`href`) of the link. |

## Example

```python
import pandas as pd

fc_articles = pd.read_csv("fc_articles.csv")	# Total of 553 fact-checking websites
fc_links = pd.read_csv("fc_links.csv")	# Total of 14,915 links
```
