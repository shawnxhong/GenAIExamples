import os
import struct
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from sklearn.cluster import KMeans
import os
from dataclasses import dataclass
from collections import deque
import heapq
import sys
import threading
import time
import logging
import operator
import os
import pickle
import uuid
import warnings
from pathlib import Path
import base64
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sized,
    Tuple,
    Union,
    TypeVar,
    Type,
)
import logging
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables.config import run_in_executor
from langchain_core.vectorstores import VectorStore

from langchain_community.docstore.base import AddableMixin, Docstore
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores.utils import (
    DistanceStrategy,
    maximal_marginal_relevance,
)
import subprocess
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_loaders import PyPDFLoader, DirectoryLoader
# from langchain.embeddings import HuggingFaceEmbeddings
from hsmlib.client_grpc import HSMClient


DEFAULT_K = 4  # Number of Documents to return.
logger = logging.getLogger(__name__)

VST = TypeVar("VST", bound="VectorStore")

def _len_check_if_sized(x: Any, y: Any, x_name: str, y_name: str) -> None:
    if isinstance(x, Sized) and isinstance(y, Sized) and len(x) != len(y):
        raise ValueError(
            f"{x_name} and {y_name} expected to be equal length but "
            f"len({x_name})={len(x)} and len({y_name})={len(y)}"
        )
    return

def is_dir_empty(dir_path):
    if not os.path.isdir(dir_path):
        raise ValueError(f"{dir_path} not exist")

    if not os.listdir(dir_path):
        return True
    else:
        return False
    
def is_index_file_present(dir_path):
    if not os.path.isdir(dir_path):
        raise ValueError(f"{dir_path} not exist")
    
    index_file = os.path.join(dir_path, "index.pkl")
    return os.path.isfile(index_file)


class HSM(VectorStore):

    _LANGCHAIN_DEFAULT_COLLECTION_NAME = "langchain"
    def __init__(
        self,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        embedding_function: Optional[Embeddings] = None,
        persist_directory: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[str] = None,
        docstore: Optional[Docstore] = None,
        index_to_docstore_id: Optional[Dict[int, str]] = None,
        # collection_metadata: Optional[Dict] = None,
        cache_size: Optional[int] = 400,
        client: Optional[HSMClient] = None,
        build_param: Optional[Dict[str,str]] = None
    ):
        """Initialize with necessary components."""
        # self.persist_directory = persist_directory
        # self.embedding_function = embedding_function
        # self.collection_name = collection_name
        # self.cur_idx = 0
        # self.cache_size = cache_size
        # self.cur_cache = 0
        # self.build_param = build_param
        # if docstore==None:
        #     self.docstore =InMemoryDocstore()
        # else:
        #     self.docstore = docstore
        # if index_to_docstore_id == None:
        #     self.index_to_docstore_id = {}
        # else:
        #     self.index_to_docstore_id = index_to_docstore_id
        # if client is not None:
        #     self._client = client
        # else:
        #     self._client = HSMClient(host=host,port=port)
        # text = "This is a sample text."
        # ebd = self.embedding_function.embed_documents([text])[0]
        # dims = len(ebd)
        # data_type = type(ebd[0]).__name__
        # # print(data_type)
        # res_code = self._client.add_table(collection_name,data_type,dims,persist_directory)
        # if res_code ==1:
        #     self.load_local(self.persist_directory,self.collection_name,self.embedding_function,host,port,"index")
        table_path = persist_directory+collection_name
        print("table name ",table_path)
        # self.load_local(persist_directory,collection_name,embedding_function,host,port,"index")
        self.persist_directory = persist_directory
        self.embedding_function = embedding_function
        self.collection_name = collection_name
        self.cache_size = cache_size
        self.cur_cache = 0
        self.build_param = build_param
        if docstore==None:
            self.docstore =InMemoryDocstore()
        else:
            self.docstore = docstore

        if index_to_docstore_id == None:
            self.index_to_docstore_id = {}
        else:
            self.index_to_docstore_id = index_to_docstore_id

        if os.path.exists(table_path) and is_index_file_present(table_path):
            print("enteeeeeeeer")
            in_path = table_path+"/index.pkl"
            print(in_path)
            # load docstore and index_to_docstore_id
            with open(in_path, "rb") as f:
                self.docstore, self.index_to_docstore_id = pickle.load(f)
        print(type(self.docstore))
        print(self.docstore._dict.keys())
        
        self.cur_idx = len(self.docstore._dict.keys())
        print(self.cur_idx)
        if client is not None:
            self._client = client
        else:
            self._client = HSMClient(host=host,port=port)
        text = "This is a sample text."
        ebd = self.embedding_function.embed_documents([text])[0]
        dims = len(ebd)
        data_type = type(ebd[0]).__name__
        # print(data_type)
        self._client.add_table(collection_name,data_type,dims,persist_directory)


    def encode_image(self, uri: str) -> str:
        """Get base64 string from image URI."""
        with open(uri, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    # def delete // todo

    @classmethod
    def from_documents(
        cls: Type[VST],
        documents: List[Document],
        embedding: Embeddings,
        persist_directory: Optional[str] = None,
        **kwargs: Any,
    ) -> VST:
        """Return VectorStore initialized from documents and embeddings."""
        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]
        return cls.from_texts(texts, embedding, metadatas=metadatas, persist_directory=persist_directory,**kwargs)

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        persist_directory: Optional[str] = None,
        # ids: Optional[List[str]] = None,
        **kwargs: Any,
    )-> int:
        HSM_collection = cls(
            collection_name=collection_name,
            embedding_function=embedding,
            persist_directory=persist_directory,
            # client_settings=client_settings,
            # client=client,
            # collection_metadata=collection_metadata,
            **kwargs,
        )
        HSM_collection.add_texts(texts=texts, metadatas=metadatas)
        return HSM_collection

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of unique IDs.

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        # ids = [str(uuid.uuid4()) for _ in texts]
        print("hsm add text")
        texts = list(texts)
        if self.embedding_function is not None:
            embeddings = self.embedding_function.embed_documents(texts)
        # print(len(embeddings))
        _len_check_if_sized(texts, metadatas, "texts", "metadatas")
        _metadatas = metadatas or ({} for _ in texts)
        documents = [
            Document(page_content=t, metadata=m) for t, m in zip(texts, _metadatas)
        ]
        if ids is None:
            ids = [i for i in range(self.cur_idx, self.cur_idx+len(texts))]
            self.cur_idx= self.cur_idx+len(texts)
        _len_check_if_sized(documents, embeddings, "documents", "embeddings")
        _len_check_if_sized(documents, ids, "documents", "ids")


        length_diff = len(texts) - len(metadatas)
        if length_diff:
            metadatas = metadatas + [{}] * length_diff

        # order id : user ids    user ids - docs
        self.docstore.add({id_: doc for id_, doc in zip(ids, documents)})
        starting_len = len(self.index_to_docstore_id)
        index_to_id = {starting_len + j: id_ for j, id_ in enumerate(ids)}
        # print(index_to_id)
        self.index_to_docstore_id.update(index_to_id)

        self._client.add_text_loop(
                    table_name=self.collection_name,
                    vector_ids=list(index_to_id.keys()),
                    np_array=embeddings,
        )
        self.cur_cache += len(ids)
        if self.cur_cache >= self.cache_size:
            self._client.build_index(self.collection_name,self.build_param)
            self.cur_cache = 0
                    
        self.save_local()
        
        return index_to_id

    def add_images(
        self,
        uris: List[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of unique IDs.

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        # ids = [str(uuid.uuid4()) for _ in texts]
        print("hsm add img")
        b64_texts = [self.encode_image(uri=uri) for uri in uris]
        if self.embedding_function is not None and hasattr(
            self.embedding_function, "embed_image"
        ):
            embeddings = self.embedding_function.embed_image(uris=uris)
        # print(len(embeddings))
        # _len_check_if_sized(texts, metadatas, "texts", "metadatas")
        _metadatas = metadatas or ({} for _ in uris)
        documents = [
            Document(page_content=t, metadata=m) for t, m in zip(b64_texts, _metadatas)
        ]
        if ids is None:
            ids = [i for i in range(self.cur_idx, self.cur_idx+len(uris))]
            self.cur_idx= self.cur_idx+len(uris)
        # _len_check_if_sized(documents, embeddings, "documents", "embeddings")
        # _len_check_if_sized(documents, ids, "documents", "ids")


        length_diff = len(uris) - len(metadatas)
        if length_diff:
            metadatas = metadatas + [{}] * length_diff

        # order id : user ids    user ids - docs
        self.docstore.add({id_: doc for id_, doc in zip(ids, documents)})
        starting_len = len(self.index_to_docstore_id)
        index_to_id = {starting_len + j: id_ for j, id_ in enumerate(ids)}
        # print(index_to_id)
        self.index_to_docstore_id.update(index_to_id)

        self._client.add_text_loop(
                    table_name=self.collection_name,
                    vector_ids=list(index_to_id.keys()),
                    np_array=embeddings,
        )
        print("add image: ",len(ids))
        # if self.cur_cache >= self.cache_size:
        #     self._client.build_index()
        #     self.cur_cache = 0  
        print("docstore after add")
        print(self.docstore._dict.keys())      
        self.save_local()
        
        return index_to_id

    def close(self, 
        table_name:str,ids: Optional[List[str]] = None, **kwargs: Any) -> None:
        """Delete by vector IDs.

        Args:
            ids: List of ids to delete.
        """
        # delete meta
        self._client.close_table_loop(table_name=table_name)

    def similarity_search(
        self, query: str, k: int = 4, Ls: int = 16, **kwargs: Any
    ) -> List[Document]:
        # q_embedding = self._embed_documents(query)
        q_embedding = self.embedding_function.embed_query(query)

        if 'collection_name' in kwargs:
            cur_collection_name = kwargs['collection_name']
        else:
            cur_collection_name = self.collection_name

        if 'debug' in kwargs:
            debugF = kwargs['debug']
        else:
            debugF = False

        indices, debug_info = self._client.query_loop(table_name=cur_collection_name,query_vs=q_embedding,Ls=Ls,k_value=k,debug=debugF)
        docs = []
        # print(indices[0])
        print("lennnnnn of result: ", len(indices))
        for i in indices[0]:
            if i == -1:
                # This happens when not enough docs are returned.
                continue
            _id = self.index_to_docstore_id[i]
            doc = self.docstore.search(_id)
            if not isinstance(doc, Document):
                raise ValueError(f"Could not find document for id {_id}, got {doc}")
            docs.append(doc)
        
        my_docs = docs[:k]
        print("-----HSM----")
        print(type(my_docs))
        print(len(my_docs))
        print(type(my_docs[0]))
        return docs[:k]
        # return docs[:k], debug_info
    
    def similarity_batch_search(
        self, query: str, k: int = 4, Ls: int = 16, **kwargs: Any
    ) -> List[Document]:
        q_embeddings = []
        for q in query:
            q_embeddings.append(self.embedding_function.embed_query(q))

        if 'collection_name' in kwargs:
            cur_collection_name = kwargs['collection_name']
        else:
            cur_collection_name = self.collection_name

        if 'debug' in kwargs:
            debugF = kwargs['debug']
        else:
            debugF = False

        indices, debug_info = self._client.query_loop(table_name=cur_collection_name,query_vs=q_embeddings,Ls=Ls,k_value=k,debug=debugF)
        res_docs = []
        for indice in indices:
            docs = []
            for i in indice:
                if i == -1:
                    continue
                _id = self.index_to_docstore_id[i]
                doc = self.docstore.search(_id)
                if not isinstance(doc, Document):
                    raise ValueError(f"Could not find document for id {_id}, got {doc}")
                docs.append(doc)
            res_docs.append(docs)

        return res_docs, debug_info

    def collection_status(self,collection_name: List[str] = None):
        self._client.table_status(collection_name)
        
    def save_local(self, index_name: str = "index") -> None:
        """Save FAISS index, docstore, and index_to_docstore_id to disk.

        Args:
            folder_path: folder path to save index, docstore,
                and index_to_docstore_id to.
            index_name: for saving with a specific index file name
        """
        path = Path(self.persist_directory)
        path.mkdir(exist_ok=True, parents=True)
        # save docstore and index_to_docstore_id
        with open(path / self.collection_name /f"{index_name}.pkl", "wb") as f:
            pickle.dump((self.docstore, self.index_to_docstore_id), f)

    @classmethod
    def load_local(
        cls,
        folder_path: str,
        table_name:str,
        embeddings: Embeddings,
        host:str,
        port:int,
        index_name: str = "index",
        **kwargs: Any,
    ) -> FAISS:
       
        path = Path(folder_path)

        # load docstore and index_to_docstore_id
        with open(path / f"{index_name}.pkl", "rb") as f:
            docstore, index_to_docstore_id = pickle.load(f)
                   
        # collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        # embedding_function: Optional[Embeddings] = None,
        # persist_directory: Optional[str] = None,
        # ip_addr: Optional[str] = None,
        # # collection_metadata: Optional[Dict] = None,
        # client: Optional[HSMClient] = None,
        # return cls(embeddings, index, docstore, index_to_docstore_id, **kwargs)
        return cls(table_name,embeddings,folder_path,host,port, docstore, index_to_docstore_id, **kwargs)
