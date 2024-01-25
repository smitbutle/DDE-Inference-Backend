<div align="center">

# Backend for Doc-Data-Extractor: Convert unstructured document to structured key-value data.

</div>

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```
3. API Endpoints:

```bash
GET /: Home page of the application.
POST /upload: Endpoint to upload images and process them.
```

## Use cases

Extracting key-value pairs out of the unstructured business documents like scanned bill-of-lading.

![](https://blogs.sap.com/wp-content/uploads/2020/11/fig12.png)

Information extracted from the document image along with the location of the extracts 

_(Not actual frontend, Image just for representation purpose)_


## Limitations

Handwritten invoices should be avoided, because of factors such as Variability in Handwriting, Lack of Standardization, Irregularities and Artifacts and Cursive Writing.


