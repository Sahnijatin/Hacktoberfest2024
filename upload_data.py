import os
from tika import parser
from langchain_community.graphs import Neo4jGraph
from langchain_community.graphs.graph_document import (
    Node as BaseNode,
    Relationship as BaseRelationship,
    GraphDocument,
)
from langchain.schema import Document
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import TokenTextSplitter
from langchain.chains.openai_functions import create_structured_output_chain
from tqdm import tqdm
from typing import List, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Property(BaseModel):
    key: str = Field(..., description="key")
    value: str = Field(..., description="value")

class Node(BaseNode):
    properties: Optional[List[Property]] = Field(None, description="List of node properties")

class Relationship(BaseRelationship):
    properties: Optional[List[Property]] = Field(None, description="List of relationship properties")

class KnowledgeGraph(BaseModel):
    nodes: List[Node] = Field(..., description="List of nodes in the knowledge graph")
    rels: List[Relationship] = Field(..., description="List of relationships in the knowledge graph")

def format_property_key(s: str) -> str:
    """Format property key to camelCase"""
    try:
        if not s:
            return "unknownKey"
        words = str(s).split()
        if not words:
            return s.lower()
        first_word = words[0].lower()
        capitalized_words = [word.capitalize() for word in words[1:]]
        return "".join([first_word] + capitalized_words)
    except Exception as e:
        logger.error(f"Error formatting property key: {str(e)}")
        return "errorKey"

def props_to_dict(props) -> dict:
    """Convert properties to dictionary with error handling"""
    try:
        properties = {}
        if not props:
            return properties
            
        for p in props:
            if hasattr(p, 'key') and hasattr(p, 'value'):
                key = format_property_key(str(p.key))
                properties[key] = str(p.value)
                
        return properties
    except Exception as e:
        logger.error(f"Error converting properties to dict: {str(e)}")
        return {}

def map_to_base_node(node: Node) -> BaseNode:
    """Map Node to BaseNode with error handling"""
    try:
        properties = props_to_dict(node.properties) if node.properties else {}
        node_id = str(node.id) if node.id else "unknown"
        node_type = str(node.type) if node.type else "unknown"
        properties["name"] = node_id.title()
        
        return BaseNode(
            id=node_id.title(),
            type=node_type.capitalize(),
            properties=properties
        )
    except Exception as e:
        logger.error(f"Error mapping to base node: {str(e)}")
        raise

def map_to_base_relationship(rel: Relationship) -> BaseRelationship:
    """Map Relationship to BaseRelationship with error handling"""
    try:
        source = map_to_base_node(rel.source)
        target = map_to_base_node(rel.target)
        properties = props_to_dict(rel.properties) if rel.properties else {}
        
        return BaseRelationship(
            source=source,
            target=target,
            type=rel.type if rel.type else "relatedTo",
            properties=properties
        )
    except Exception as e:
        logger.error(f"Error mapping to base relationship: {str(e)}")
        raise

def get_extraction_chain(llm: ChatOpenAI, allowed_nodes: Optional[List[str]] = None, allowed_rels: Optional[List[str]] = None):
    """Create extraction chain with improved prompt"""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
            "system",
            f"""# Knowledge Graph Instructions for GPT-4
            ## 1. Overview
            You are a top-tier algorithm designed for extracting information in structured formats to build a knowledge graph.
            - **Nodes** represent entities and concepts.
            - The aim is to achieve simplicity and clarity in the knowledge graph.
            ## 2. Labeling Nodes
            - **Consistency**: Use basic or elementary types for node labels.
            - **Node IDs**: Use names or human-readable identifiers from the text, never integers.
            {'- **Allowed Node Labels:**' + ", ".join(allowed_nodes) if allowed_nodes else ""}
            {'- **Allowed Relationship Types**:' + ", ".join(allowed_rels) if allowed_rels else ""}
            ## 3. Handling Data
            - Attach numerical data and dates as properties of nodes.
            - **Property Format**: Use key-value format.
            - **Naming Convention**: Use camelCase for property keys.
            ## 4. Quality Requirements
            - Ensure all relationships connect existing nodes.
            - Verify all properties have valid values.
            - Maintain consistency in entity references.
            """
            ),
            ("human", "Extract information from this text into a knowledge graph: {input}"),
        ]
    )
    return create_structured_output_chain(KnowledgeGraph, llm, prompt, verbose=True)

def extract_and_store_graph(document: Document, llm: ChatOpenAI, nodes: Optional[List[str]] = None, rels: Optional[List[str]] = None) -> None:
    """Extract and store graph with comprehensive error handling"""
    try:
        extract_chain = get_extraction_chain(llm, nodes, rels)
        result = extract_chain.invoke({"input": document.page_content})
        
        logger.info(f"Extraction result structure: {type(result)}")
        
        # Handle different return types
        if isinstance(result, dict):
            knowledge_graph = result.get('function', result)
        else:
            knowledge_graph = result
            
        if not hasattr(knowledge_graph, 'nodes') or not hasattr(knowledge_graph, 'rels'):
            logger.warning("Invalid knowledge graph structure received")
            return
            
        # Process nodes
        entity_map = {}
        for node in knowledge_graph.nodes:
            try:
                base_node = map_to_base_node(node)
                entity_map[node.id] = base_node
            except Exception as e:
                logger.error(f"Error processing node {node}: {str(e)}")
                continue

        # Process relationships
        relationships = []
        for rel in knowledge_graph.rels:
            try:
                source_node = entity_map.get(rel.source.id)
                target_node = entity_map.get(rel.target.id)
                
                if source_node and target_node:
                    relationship = map_to_base_relationship(rel)
                    relationships.append(relationship)
            except Exception as e:
                logger.error(f"Error processing relationship {rel}: {str(e)}")
                continue

        # Create and store graph document
        if entity_map and relationships:
            graph_document = GraphDocument(
                nodes=list(entity_map.values()),
                relationships=relationships,
                source=document
            )
            graph.add_graph_documents([graph_document])
            logger.info(f"Successfully added graph document with {len(entity_map)} nodes and {len(relationships)} relationships")
        else:
            logger.warning("No valid nodes or relationships extracted")
            
    except Exception as e:
        logger.error(f"Error in extraction process: {str(e)}")
        raise

def main():
    """Main execution function with error handling"""
    try:
        # Configuration
        url = "bolt://localhost:7687"
        username = "neo4j"
        password = os.getenv("NEO4J_PASSWORD")
        openai_api_key = os.getenv("OPENAI_API_KEY")

        # Initialize models and graph
        global graph  # Make graph available to other functions
        llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=openai_api_key)
        graph = Neo4jGraph(url=url, username=username, password=password)
        logger.info("Successfully initialized LLM and Neo4j connection")

        # Parse PDF document
        pdf_path = r'D:\\Python Scripts AI\\Knowledge Graph QnA\\Advanced RAG\\data\\ITSM.pdf'
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        parsed_text = parser.from_file(pdf_path)
        if not parsed_text.get("content"):
            raise ValueError("Failed to parse PDF or no content found")

        # Split text into manageable chunks
        text_splitter = TokenTextSplitter(chunk_size=2000, chunk_overlap=24)
        documents = text_splitter.create_documents([parsed_text["content"]])
        logger.info(f"Successfully split document into {len(documents)} chunks")

        # Process each document chunk
        for doc in tqdm(documents, desc="Processing documents"):
            try:
                extract_and_store_graph(doc, llm)
            except Exception as e:
                logger.error(f"Error processing document chunk: {str(e)}")
                continue

        logger.info("Knowledge graph extraction completed successfully")

    except Exception as e:
        logger.error(f"Fatal error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
