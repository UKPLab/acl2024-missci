# The <span style="font-variant:small-caps;">Missci</span> Dataset
We provide a validation split and a test split for the <span style="font-variant:small-caps;">Missci</span> dataset to allow prompt engineering without compromising the unseen test set.
Both files are provided as `.jsonl` format.

## Argument Structure in the <span style="font-variant:small-caps;">Missci</span> Dataset
Each line in the files constitutes one argument. Each argument is provided using JSON formatting as below:
````json
{
    "id": "<a unique identifier>",
    "meta": {
        "fc_article_url": "<some information about the fact-checking article such as date, url, ...>",
        "...": "..."
    },
    "argument": {
        "claim": "<The claim of the argument that misrepresents a scientific publication>",
        "hidden_premises": [
            "<optional hidden premises that are accurate but not part of the scientific publication>"
        ],
        "accurate_premise_p0": {
            "premise": "<The accurate premise p0>"
        },
        "fallacies": [
            {
                "consolidation_comment": "<annotator's summary of the fallacies>",
                "fallacy_context": "<fallacy context si. This may be an empty string if p0 contains all the information to detect the fallacies>",
                "id": "<unique identifier for this fallacy, including all interchangeable fallacies>",
                "interchangeable_fallacies": [
                    {
                        "premise": "<fallacious premise 1>",
                        "class": "<applied fallacy class 1>",
                        "id": "<unique identifier for this interchangeable fallacy>"
                    },
                    {
                        "...": "<optionally additional interchangeable fallacies>"
                    }
                ]
            },
            {
                "...": "<optionally additional fallacies>"
            }
        ]
    },
    "study": {
        "url": "<url to the misrepresented scientific publication>"
    }
}
````


Each argument contains 
* a single inaccurate *claim* based on 
* a single *accurate premise* $p_0$ and
* at least one *fallacy* $F_i$ containing a *fallacy context* $s_i$ and at least one *interchangeable fallacy* of
  * a *fallacious premise* 
  * the applied *fallacy class.*
