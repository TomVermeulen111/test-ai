import re
import requests
import xml.etree.ElementTree as ET

def validate_partner_key(partnerKey: str):
    if "apikey" in partnerKey:   
        apikeyRemover = re.compile(r'apikey\w*')
        partnerKey = apikeyRemover.sub('', partnerKey)
    isValid = False
    try:
        url="https://dev-service.immo-connect.be/soap12"
        #headers = {'content-type': 'application/soap+xml'}
        headers = {'content-type': 'text/xml'}
        body = """<soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
            <soap12:Body>
                <ValidateApiKeys xmlns:i="http://www.w3.org/2001/XMLSchema-instance" xmlns="http://schemas.servicestack.net/types">
                    <PartnerApiKey>"""+ partnerKey +"""</PartnerApiKey>
                    <Session>00000000-0000-0000-0000-000000000000</Session>
                    <CustomerApiKey>00000000-0000-0000-0000-000000000000</CustomerApiKey>
                </ValidateApiKeys>
            </soap12:Body>
        </soap12:Envelope>"""

        response = requests.post(url,data=body,headers=headers)

        responseXml = ET.fromstring(response.text)

        isValid = responseXml.find('{http://www.w3.org/2003/05/soap-envelope}Body').find('{http://schemas.servicestack.net/types}ValidateApiKeysResponse').find('{http://schemas.servicestack.net/types}IsValid').text == 'true'
    except Exception as e:
        print(e)
    return isValid
