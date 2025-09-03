from fastapi import HTTPException
from httpx import AsyncClient, HTTPStatusError, TimeoutException
from core.logging_config import get_logger


from schemas.data_product import BusinessAssetCreate, BusinessObject, AssociationCreate, Association

logger = get_logger(__name__)
BASE_URL = 'http://51.103.210.156:8080/erwinDISuite'
HEADERS = {'Content-Type': 'application/json',
           'Authorization': '75a9c605-8636-4581-859a-fb14a006d99c'}


async def create_business_asset(business_object: list[BusinessObject]):
    try:
        business_create = BusinessAssetCreate(businessObjects=business_object)
        async with AsyncClient(base_url=BASE_URL, headers=HEADERS) as client:
            res = await client.post('/api/businessglossarymanager/assets', json=business_create.model_dump())
            res.raise_for_status()
            data = res.json()
            logger.info(f'Business Asset Created: {data}')
            return data
    except TimeoutException as e:
        logger.error(f'Timeout error: {e}')
        raise HTTPException(500, detail=str(e))
    except HTTPStatusError as e:
        logger.error(f'HTTP error: {e}')
        raise HTTPException(500, detail=str(e))
    except Exception as e:
        logger.error(f'Unknown error: {e}')
        raise HTTPException(500, str(e))


async def get_business_asset_id(business_asset_name: str, asset_type_name: str = 'Data Products', catalog_path: str = 'ADPC'):
    try:
        params = {
            'assetTypeName': asset_type_name,
            'businessAssetName': business_asset_name,
            'catalogPath': catalog_path
        }
        async with AsyncClient(base_url=BASE_URL, headers=HEADERS) as client:
            res = await client.get('/api/businessglossarymanager/assets/id', params=params)
            res.raise_for_status()
            data = int(res.text)
            logger.info(f'Business asset id: {data}')
            return data
    except TimeoutException as e:
        logger.error(f'Timeout error: {e}')
        raise HTTPException(500, detail=str(e))
    except HTTPStatusError as e:
        logger.error(f'HTTP error: {e}')
        raise HTTPException(500, detail=str(e))
    except Exception as e:
        logger.error(f'Unknown error: {e}')
        raise HTTPException(500, detail=str(e))


async def get_business_asset_object_type_id(business_asset_id: int):
    try:
        params = {
            'nodeType': 'BGM_CUSTOM',
            'businessAssetIds': business_asset_id
        }
        async with AsyncClient(base_url=BASE_URL, headers=HEADERS) as client:
            res = await client.get('/api/businessglossarymanager/assets', params=params)
            res.raise_for_status()
            data = res.json()['data'][0]['objectTypeId']
            logger.info(f'Business asset object type id: {data}')
            return data
    except TimeoutException as e:
        logger.error(f'Timeout error: {e}')
        raise HTTPException(500, detail=str(e))
    except HTTPStatusError as e:
        logger.error(f'HTTP error: {e}')
        raise HTTPException(500, detail=str(e))
    except Exception as e:
        logger.error(f'Unknown error: {e}')
        raise HTTPException(500, detail=str(e))


async def get_existing_table_id(system_name: str, environment_name: str, table_name: str):
    try:
        params = {
            'systemName': system_name,
            'environmentName': environment_name,
            'tableName': table_name
        }
        async with AsyncClient(base_url=BASE_URL, headers=HEADERS) as client:
            res = await client.get('/api/metadatamanager/tables/id', params=params)
            res.raise_for_status()
            data = int(res.text)
            if data < 0:
                logger.info(f'Table {table_name} does not exist')
                raise HTTPException(404, detail=f"Table {table_name} in system: {system_name} under environment: {environment_name} not found")
            logger.info(f'Existing table id: {res.text}')
            return data
    except TimeoutException as e:
        logger.error(f'Timeout error: {e}')
        raise HTTPException(500, detail=str(e))
    except HTTPStatusError as e:
        logger.error(f'HTTP error: {e}')
        raise HTTPException(500, detail=str(e))
    except Exception as e:
        logger.error(f'Unknown error: {e}')
        raise HTTPException(500, detail=str(e))


async def get_existing_table_object_type_id(table_id: int):
    try:
        params = {
            'tableIds': table_id
        }
        async with AsyncClient(base_url=BASE_URL, headers=HEADERS) as client:
            res = await client.get('/api/metadatamanager/tables', params=params)
            res.raise_for_status()
            data = res.json()['data'][0]['objectTypeId']
            logger.info(f'Business asset object type id: {data}')
            return data
    except TimeoutException as e:
        logger.error(f'Timeout error: {e}')
        raise HTTPException(500, detail=str(e))
    except HTTPStatusError as e:
        logger.error(f'HTTP error: {e}')
        raise HTTPException(500, detail=str(e))
    except Exception as e:
        logger.error(f'Unknown error: {e}')
        raise HTTPException(500, detail=str(e))


async def create_association(association: list[Association]):
    try:
        association_create = AssociationCreate(associations=association)
        async with AsyncClient(base_url=BASE_URL, headers=HEADERS) as client:
            res = await client.post('/api/miscellaneous/associations', json=association_create.model_dump())
            res.raise_for_status()
            data = res.json()
            logger.info(f'Association Created: {data}')
            return data
    except TimeoutException as e:
        logger.error(f'Timeout error: {e}')
        raise HTTPException(500, detail=str(e))
    except HTTPStatusError as e:
        logger.error(f'HTTP error: {e}')
        raise HTTPException(e.response.status_code, detail=str(e.response.text))
    except Exception as e:
        logger.error(f'Unknown error: {e}')
        raise HTTPException(500, detail=str(e))





