from dotenv import load_dotenv
import os
from langchain_openai import AzureChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import streamlit as st
from typing import Any, Dict, List
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import BaseMessage
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings
from langchain.callbacks.base import BaseCallbackHandler

class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""

    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Add a list of messages to the store"""
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []

class CustomHandler(BaseCallbackHandler):

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        formatted_prompts = "\n".join(prompts)
        st.session_state["last_generated_prompt"]=formatted_prompts
        print(formatted_prompts)

# https://python.langchain.com/v0.1/docs/use_cases/question_answering/chat_history/

load_dotenv()

index_name="demo-index-coman"

llm = AzureChatOpenAI(
    openai_api_version=str(os.getenv("AZURE_OPENAI_API_VERSION")),
    azure_deployment=str(os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")),
)

AZURE_SEARCH_KEY = str(os.getenv("AZURE_SEARCH_KEY"))

def get_filter_for_context(context):
    if context =="CIB-lid":
        return 'source eq "123"'
    elif context == "Niet CIB-lid":
        return None
    elif context == "Syllabusverbod":
        return None

def create_conversational_rag_chain(system_prompt, context, nr_of_docs_to_retrieve):
    
    ### Contextualize question ###
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
   
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment="orisai-text-embedding-3-large-development",
    )

    vector_store: AzureSearch = AzureSearch(
        azure_search_endpoint=str(os.getenv("BASE_URL")),
        azure_search_key=AZURE_SEARCH_KEY,
        index_name=index_name,
        embedding_function=embeddings.embed_query
    )

    retriever = vector_store.as_retriever(k=nr_of_docs_to_retrieve)
                                                         
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    ### Answer question ###
    qa_system_prompt = system_prompt + """"

    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


    ### Statefully manage chat history ###
    if "store" not in st.session_state:
        st.session_state["store"] = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in st.session_state["store"]:
            st.session_state["store"][session_id] = InMemoryHistory()
        return st.session_state["store"][session_id]
    

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
        **{"filters": "source eq '1234'"}
    )
    return conversational_rag_chain


def document_data(conversational_rag_chain: RunnableWithMessageHistory, query):    
    return conversational_rag_chain.invoke(
        {"input": query},
        config={"configurable": {"session_id": "abc123"},"callbacks": [CustomHandler()]},
    )   
    
if __name__ == '__main__':

    if "user_prompt_history" not in st.session_state:
       st.session_state["user_prompt_history"]=[]
    if "chat_answers_history" not in st.session_state:
       st.session_state["chat_answers_history"]=[]
    if "last_generated_prompt" not in st.session_state:
       st.session_state["last_generated_prompt"]='test'

    st.header("QA ChatBot")
    # ChatInput
    prompt = st.chat_input("Enter your questions here")

    with st.sidebar:
        system_prompt = st.text_area(value="""You are an assistant for question-answering tasks. 
                                     
You can use the following pieces of retrieved context to answer the question. 
                                     
Use three sentences maximum and keep the answer concise.
                                     
You will have a chat history, but you must only answer the last question.
                                     
You MUST answer in dutch.""", label="Systeem prompt", height=275
, help="""Eerst wordt gezocht naar de x (hieronder te configureren) best matchende documenten in de vector store. 
Vervolgens wordt deze systeem prompt, samen met de inhoud van die documenten naar de llm gestuurd om een antwoord te genereren
""")
        nr_of_docs_to_retrieve = st.number_input(value=3, label="Aantal documenten die meegestuurd worden", min_value=1,
            help="Aantal documenten die opgehaald worden uit de vector store en meegestuurd worden naar de llm")

        # context = st.selectbox("Selecteer je context", options=["CIB-lid", "Niet CIB-lid", "Syllabusverbod"], help="""
        #              CIB-Lid: Toegang tot alles\n
        #              Niet CIB-lid: Enkel toegang tot publieke zaken (geen bijlages)\n
        #              Syllabusverbod: Toegang tot alles behalve syllabi\n
        #              """)

        context = "CIB-Lid"
        
        last_generated_prompt_text_area = st.text_area("Voorbeeld van een prompt", height=275, value="""System: You are an assistant for question-answering tasks.

You can use the following pieces of retrieved context to answer the question.

Use three sentences maximum and keep the answer concise.

You will have a chat history, but you must only answer the last question.

You MUST answer in dutch."

    <p>Vanaf 23 november 2022 zal er een informatieplicht gelden bij de verkoop van een pand met bouwjaar 2000 of ouder. Die zal gelden voor elke overdracht waarvan het compromis ondertekend wordt vanaf 23 november 2022. De inhoud van het attest moet bekendgemaakt worden aan de koper van het pand bij het ondertekenen van het compromis. Vanaf 2032 moet elke eigenaar van een pand dat gebouwd is voor 2001 beschikken over een asbestattest. Maar wat als je de informatieplicht niet naleeft? Mag je in dat geval sancties verwachten?</p> <ol><li><strong>Context</strong></li></ol><p>De nieuw ingevoerde bijzondere informatieverplichting rond de aanwezigheid van asbest laat zich rechtvaardigen door het bewezen grote gezondheidsrisico van asbesthoudende materialen en het gebrek aan voldoende kennis rond asbest van de gemiddelde koper van onroerend goed.</p><p>Een verplichting om te informeren, zelfs al is deze breedvoerig publiek gemaakt en zelfs al ziet nagenoeg iedereen er de legitimiteit van in, is maar zo adequaat als een efficiënte handhaving van haar naleving mogelijk is. De vraag stelt zich of er aan het niet naleven van deze informatieverplichting al dan niet sancties verbonden zijn.&nbsp;</p><p>&nbsp;</p><p><strong>2. Privaatrechtelijke sanctie</strong></p><p>De verkopende eigenaar moet de inhoud van en geldig asbestattest al meedelen aan de koper n.a.v. het aangaan van de overdracht. Bovendien heeft hij de verplichting om in de onderhandse verkoopovereenkomst te vermelden of het asbestattest voorafgaand aan de ondertekening aan de koper is medegedeeld. Het onderhandse document vermeldt ook de datum van het asbestattest, de samenvattende conclusie en de unieke code.</p><p>Voorts moet ook elke authentieke overdrachtsakte er melding van maken of het asbestattest tijdig meegedeeld is aan de koper. Deze akte neemt bovendien de datum, de samenvattende conclusie en de unieke code van het asbestattest in de overwegingen op.</p><p>Naar analogie met wat het geval is bij miskenning van de bijzondere informatieverplichtingen m.b.t. het bodemattest en het stedenbouwkundige uittreksel, voorziet de decreetgever ook in een expliciete en bijzondere nietigheidssanctie voor de gevallen waarin een verkoper de verplichtingen inzake het asbestattest niet nakomt.</p><p>Art. 33/14, § 6 van het Materialendecreet &nbsp;bepaalt: “<i><strong>De verwerver kan de nietigheid vorderen van de overdracht die heeft plaatsgevonden in strijd met”&nbsp;</strong></i>de bijzondere informatieverplichtingen ter zake het asbestattest.</p><p>Het is een relatieve nietigheid. In dit opzicht voorziet art. 33/14, § 6, tweede lid van het Materialendecreet:</p><p>“<i><strong>De nietigheid kan niet meer worden ingeroepen als de verwerver zijn verzaking van de nietigheidsvordering uitdrukkelijk in de authentieke akte heeft laten opnemen en hij een geldig asbestattest heeft gekregen.</strong></i>”</p><p>&nbsp;</p><p><strong>3. Deontologisch en bestuursrechtelijk&nbsp;</strong></p><p>Weet ook dat de niet-naleving van de informatieverplichting door vastgoedmakelaar tuchtrechtelijke gevolgen kan hebben.</p><p>Daarenboven kan elk verzuim van de verplichtingen ook aanleiding kunnen geven tot een bestuurlijke geldboete.</p><p>Bij het opleggen van die boetes wordt rekening gehouden met zowel technische kenmerken van de inbreuk(locatie), de ernst ervan, als de persoonlijkheid en de antecedenten van de inbreukmaker.&nbsp;</p> <p>Vanaf 23 november 2022 zal er een informatieplicht gelden bij de verkoop van een pand met bouwjaar 2000 of ouder. Die zal gelden voor elke overdracht waarvan het compromis ondertekend wordt vanaf 23 november 2022. De inhoud van het attest moet bekendgemaakt worden aan de koper van het pand bij het ondertekenen van het compromis. Vanaf 2032 moet elke eigenaar van een pand dat gebouwd is voor 2001 beschikken over een asbestattest. Maar wat als je de informatieplicht niet naleeft? Mag je in dat geval sancties verwachten?</p> Zijn er sancties gekoppeld aan het niet naleven van de asbestattestverplichting?

Wanneer is een asbestattest verplicht? <p>Voor het verplicht beschikken over een asbestinventarisattest, of kortweg asbestattest, wordt een tweetrapsraket gebruikt. In eerste instantie wordt gewerkt via een informatieplicht bij verkoop vanaf 23 november en daarnaast geldt een algemene verplichting tegen 31 december 2031.</p> <p>Een cruciaal element waar voor beide verplichtingen rekening mee moet worden gehouden is het criterium ‘toegankelijke constructies met risicobouwjaar’.&nbsp; Onder het risicobouwjaar wordt verstaan: bouwjaar 2000 of ouder. Concreet is het v<strong>erplichte asbestattest niet van toepassing op panden met een bouwjaar vanaf 2001 en recenter</strong>. Behoudens bewijs van het tegendeel geldt het jaar van opname in het kadaster als bouwjaar.</p><p><strong>Belangrijk: de verplichtingen zijn niet beperkt tot residentieel vastgoed! Ze gelden ook voor winkels, horeca, kantoren, industriegebouwen, …</strong></p><h3><strong>Informatieplicht bij verkoop&nbsp;</strong></h3><p>Vanaf 23 november 2022 zal er een informatieplicht gelden bij de verkoop van een constructie met bouwjaar 2000 of ouder. Idem bij de vestiging of de overdracht van een recht van vruchtgebruik, een recht van erfpacht, een opstalrecht of een zakelijk recht van gebruik. Erfenissen, schenkingen en onteigeningen zijn evenwel niet onderworpen aan de informatieplicht.</p><p><i><strong>Onderhandse verkoopovereenkomst</strong></i></p><p><strong>Concreet moet de verkoper bij het sluiten van een onderhandse verkoopovereenkomst aan de kandidaat-koper de inhoud meedelen van een geldig asbestattest</strong>. De onderhandse verkoopovereenkomst moet daartoe het volgende vermelden: de datum van het attest, de samenvattende conclusie, de unieke code van het attest en een clausule die stelt dat de inhoud van dit (geldige) attest voorafgaandelijk is medegedeeld aan de koper, of, indien men hierin in gebreke is gebleven, dat dit niet is gebeurd.</p><p><i><strong>Appartement</strong></i></p><p>Betreft het een appartement of een kavel binnen een gebouw in mede-eigendom, dan moeten er twee asbestattesten voorgelegd worden: (1) dat van de gemeenschappelijke delen en (2) dat van de individuele kavel.&nbsp;</p><p><strong>Voor de gemeenschappelijke delen gelden de verplichting en de navenante informatieplicht evenwel pas vanaf 1/01/2025</strong>. Tot dan volstaat de aanwezigheid van een asbestattest voor het private deel bij overdracht.&nbsp;</p><p>Vanaf 2025 zal je zowel het attest voor het appartement zelf als het attest voor het gebouw aan een koper moeten overhandigen en breiden dus ook de vermeldingen in de onderhandse verkoopovereenkomst uit.</p><p><i><strong>Niet in publiciteit</strong></i></p><p>Het Materialendecreet maakt geen melding van een informatieplicht in de publiciteit. Er zullen dus geen extra bijzondere vermeldingen verplicht zijn in immo-advertenties, zij het online of op de verkoopborden aan het pand zelf. De informatieplicht focust op het moment van/voor de ondertekening van de onderhandse verkoopovereenkomst.</p><p><i><strong>Nietigheid</strong></i></p><p>Wordt de informatieplicht niet nageleefd dan kan de koper de nietigheid van de zakenrechtelijke transactie eisen, tenzij de koper in de authentieke akte aan deze mogelijkheid heeft verzaakt en hij/zij intussen een geldig attest heeft ontvangen.</p><h3><strong>Algemene verplichting tegen 31 december 2031</strong></h3><p>Op 31 december 2031 moet elke eigenaar van een constructie met bouwjaar 2000 of ouder over een geldig asbestattest beschikken. Betreft het een appartement of een kavel binnen een mede-eigendom, dan moeten er twee attesten zijn en dus ook één voor de gemeenschappelijke delen. Tegen 31 december 2031 zal er dus ook moet er sowieso een asbestattest voor de gemeenschappelijke delen zijn.</p>

<p><strong>Inventariseren, vaststellen en beschermen, wat zijn de verschillen?</strong><br />Geïnventariseerd onroerend goed is opgenomen in een wetenschappelijke inventaris. Zo&rsquo;n opname heeft geen rechtsgevolgen. Het goed wordt enkel beschreven en gedocumenteerd.</p><p>Vastgesteld onroerend erfgoed is opgenomen in een inventaris én via een juridische procedure &lsquo;vastgesteld&rsquo;. Bij een vastgesteld item moet de overheid, eigenaar of beheerder rekening houden met bepaalde rechtsgevolgen, die verschillen naar gelang de inventaris.</p><p>Aan een bescherming is een andere procedure gekoppeld, met andere rechtsgevolgen. Onroerend erfgoed wordt beschermd omdat het van grote waarde is voor de gemeenschap. Het moet minimaal in de staat blijven waarin het zich bevond op het moment van de bescherming.</p><p><strong>Informatieverplichtingen voor onroerende goederen die zijn opgenomen in een vastgestelde inventaris</strong><br />Is een onroerend goed opgenomen in een vastgestelde inventaris, dan geldt een informatieplicht voor de volgende rechtshandelingen; verkoop, verhuur voor meer dan 9 jaar, inbrengen in vennootschap, overdracht of vestiging erfpacht of opstal, elke andere eigendomsoverdracht.</p><p>Volgende concrete verplichtingen en vermeldingen dienen nageleefd te worden;</p><ol><li>In de onderhandse en authentieke akte moet vermeld worden dat het gaat om een onroerend goed dat is opgenomen in een vastgestelde inventaris en moet verwezen worden naar de rechtsgevolgen van een inventarisitem meer bepaald hoofdstuk 4 van het Onroerenderfgoeddecreet.</li><li>De notaris die de onderhandse&nbsp; overeenkomst overneemt in een authentieke akte moet bovendien nagaan of in de onderhandse akte de bovenvermelde gegevens zijn opgenomen. Als dat niet is gebeurd, wijst de notaris de partijen hierop.</li><li>Het niet naleven van de informatieplichten wordt als inbreuk strafbaar gesteld met een exclusieve bestuurlijke geldboete van maximaal 10.000 euro.</li></ol><p><strong>Informatieverplichtingen voor beschermde goederen</strong><br />Wanneer een onroerend goed voorlopig of definitief beschermd is als monument, stads-of dorpsgezicht, (cultuurhistorisch) landschap of archeologische site of zone dan geldt de informatieplicht voor de volgende rechtshandelingen: verkoop, verhuur voor meer dan 9 jaar, inbrengen in vennootschap, overdracht of vestiging erfpacht of opstal, elke andere eigendomsoverdracht</p><p>Volgende concrete verplichtingen en vermeldingen dienen nageleefd te worden;</p><ol><li>In de publiciteit errond moet vermeld worden dat het gaat om een beschermd goed en wat de rechtsgevolgen zijn die verbonden zijn aan een bescherming.</li><li>In de onderhandse en authentieke akte moet vermeld worden dat het gaat om een beschermd goed en moet verwezen worden naar het (voorlopige of definitieve) beschermingsbesluit en de rechtsgevolgen van bescherming (hoofdstuk 6 van het Onroerenderfgoeddecreet).</li><li>De notaris die de onderhandse&nbsp; overeenkomst overneemt in een authentieke akte moet bovendien nagaan of in de onderhandse akte de bovenvermelde gegevens zijn opgenomen. Als dat niet is gebeurd, wijst de notaris de partijen hierop.</li><li>Het niet naleven van de informatieplichten wordt strafbaar gesteld met een exclusieve bestuurlijke geldboete van maximaal 10.000 euro.</li></ol> <p>Indien je als vastgoedmakelaar in aanraking komt met onroerend erfgoed moet je bepaalde informatieverplichtingen naleven. Het is hierbij belangrijk goed het onderscheid te kennen tussen onroerende goederen die zijn opgenomen in een vastgestelde inventaris en beschermde goederen.</p> <p>Indien je als vastgoedmakelaar in aanraking komt met onroerend erfgoed moet je bepaalde informatieverplichtingen naleven. Het is hierbij belangrijk goed het onderscheid te kennen tussen onroerende goederen die zijn opgenomen in een vastgestelde inventaris en beschermde goederen.</p> Welke informatieverplichtingen moet de vastgoedmakelaar naleven rond onroerend erfgoed?
Human: Zijn er sancties gekoppeld aan het niet naleven van de asbestattestverplichting?
System: Given a chat history and the latest user question     which might reference context in the chat history, formulate a standalone question     which can be understood without the chat history. Do NOT answer the question,     just reformulate it if needed and otherwise return it as is.
Human: Zijn er sancties gekoppeld aan het niet naleven van de asbestattestverplichting?
AI: Ja, er zijn sancties verbonden aan het niet naleven van de asbestattestverplichting. Bij niet-naleving kan de koper de nietigheid van de zakenrechtelijke transactie eisen, tenzij de koper in de authentieke akte aan deze mogelijkheid heeft verzaakt en een geldig attest heeft ontvangen. Daarnaast kan het niet naleven van de verplichtingen ook leiden tot tuchtrechtelijke gevolgen voor vastgoedmakelaars en bestuurlijke geldboetes.
Human: sinds wanneer is dit?""")

    if "user_prompt_history" not in st.session_state:
       st.session_state["user_prompt_history"]=[]
    if "chat_answers_history" not in st.session_state:
       st.session_state["chat_answers_history"]=[]

    if prompt:
       with st.spinner("Generating......"):
            chain = create_conversational_rag_chain(system_prompt=system_prompt, context=context, nr_of_docs_to_retrieve=nr_of_docs_to_retrieve)
            output=document_data(query=prompt, conversational_rag_chain=chain)

          # Storing the questions, answers and chat history
            answer = output['answer']
            sources = []
            for c in output['context']:
                if c.metadata['type'] == "Actua":                    
                    sources.append(c.metadata['type'] + ": https://cib-website-development.azurewebsites.net/actua/" + c.metadata['source'] + "/blabla")
                else:
                    sources.append(c.metadata['type'] + ": https://cib-website-development.azurewebsites.net/kennis/" + c.metadata['source'] + "/blabla")
            if(len(sources) > 0):
                answer += """

                Bronnen:
    """ 
                answer += "\n\t".join(sources)
            st.session_state["user_prompt_history"].append(prompt)
            st.session_state["chat_answers_history"].append(answer)

    # Displaying the chat history

    if st.session_state["chat_answers_history"]:
       for i, j in zip(st.session_state["chat_answers_history"],st.session_state["user_prompt_history"]):
          message1 = st.chat_message("user")
          message1.write(j)
          message2 = st.chat_message("assistant")
          message2.write(i)