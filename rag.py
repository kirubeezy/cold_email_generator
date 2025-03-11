from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key="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",
    temperature=0.9,
    
    # other params...
)

response = llm.invoke("What is the capital of India?")
print(response.content)
response = llm.invoke("how many grammy has jay z won? and list them with years")
print(response.content)

response = llm.invoke("the first person to walk on the moon")
print(response.content)

from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader(
    web_path = "https://www.amazon.jobs/en/jobs/2870687/senior-data-scientist-amazon-pay")
page_data = loader.load().pop().page_content
print(page_data)

# scraped text from website:

from langchain_core.prompts import PromptTemplate
prompt_extract = PromptTemplate.from_template(
        """
        ###scraped text from website:
        {page_data}
        ###INSTRUCTIONS:
        the scaped text is from ther career's page of a website.
        your job is to extract the job postings and return them in JSON format containing the
        followings keys:'role','experience','skills' and 'description'.
        only return the valid JSON.
        ###valid JSON (NO PREAMBLE) 
        """
    )
chain_extract = prompt_extract | llm
response = chain_extract.invoke(input={"page_data": page_data})
print(response.content)

# checking the type of response

type(response.content)

# as we can see that its i str lets convert into JSON (so we can use JSON Parser)

from langchain_core.output_parsers import JsonOutputParser

output_parser = JsonOutputParser()

# Assuming 'response' is your LangChain response object
json_response = output_parser.parse(response.content)

print(json_response)
# Checking if its JSON Format
type(json_response)
# Now we will create a Prompt Template for cold email

linkedin_url = "https://www.linkedin.com/in/kiran-solanki-3851b4110/"
prompt_email = PromptTemplate.from_template(
    """

    ###JOB DESCRIPTIONS:
    {job_description}
    ###INSTRUCTIONS:
    you are kiran Solanki,a data scientist from sony Music Enterntainment.
    over my experience, I have worked with various companies and have a good understanding of the industry.
    I have extracted the job postings from the previous step and need to send them to the email address provided below.
    your job is to write an cold email to the manager/recruiter of the company regarding the job mention above describing the capability
    in the fulfilling their needs.
    also add the most relevent experience and skills that you have.
    remember you are Kiran Solanki, a data scientist from sony Music Enterntainment.
    DO not provide a preamble.
    also add the {linkedin_url} profile link at the end of the email.
    ###email address(NO PREAMBLE):
    """
)
chain_email = prompt_email | llm
res = chain_email.invoke(input={"job_description":str("job"), "linkedin_url":linkedin_url})
print(res.content)