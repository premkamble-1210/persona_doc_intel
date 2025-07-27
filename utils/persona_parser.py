"""
Persona and job-to-be-done configuration parser.
Handles loading and validation of persona definitions for document intelligence.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml

logger = logging.getLogger(__name__)


class PersonaValidationError(Exception):
    """Custom exception for persona validation errors."""
    pass


class PersonaParser:
    """
    Parser for persona and job-to-be-done configurations.
    Supports JSON and YAML formats with validation.
    """
    
    def __init__(self):
        self.required_persona_fields = ['role', 'description', 'expertise_areas', 'priorities']
        self.required_job_fields = ['title', 'description', 'key_objectives', 'success_criteria']
        self.supported_formats = ['.json', '.yaml', '.yml']
    
    def load_from_file(self, file_path: str) -> Dict[str, Any]:
        """
        Load persona configuration from file.
        
        Args:
            file_path: Path to configuration file (JSON or YAML)
            
        Returns:
            Dictionary containing persona and job configuration
            
        Raises:
            PersonaValidationError: If configuration is invalid
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Persona file not found: {file_path}")
        
        if file_path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        logger.info(f"Loading persona configuration from: {file_path.name}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix.lower() == '.json':
                    config = json.load(f)
                else:  # YAML
                    config = yaml.safe_load(f)
            
            # Validate and normalize configuration
            validated_config = self._validate_config(config)
            
            logger.info(f"Successfully loaded persona: {validated_config['persona']['role']}")
            return validated_config
            
        except json.JSONDecodeError as e:
            raise PersonaValidationError(f"Invalid JSON format: {str(e)}")
        except yaml.YAMLError as e:
            raise PersonaValidationError(f"Invalid YAML format: {str(e)}")
        except Exception as e:
            raise PersonaValidationError(f"Failed to load persona configuration: {str(e)}")
    
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize persona configuration."""
        
        if not isinstance(config, dict):
            raise PersonaValidationError("Configuration must be a dictionary")
        
        # Check required top-level sections
        if 'persona' not in config:
            raise PersonaValidationError("Missing 'persona' section")
        if 'job_to_be_done' not in config:
            raise PersonaValidationError("Missing 'job_to_be_done' section")
        
        # Validate persona section
        persona = config['persona']
        self._validate_persona(persona)
        
        # Validate job-to-be-done section
        job = config['job_to_be_done']
        self._validate_job(job)
        
        # Normalize and enrich configuration
        normalized_config = {
            'persona': self._normalize_persona(persona),
            'job_to_be_done': self._normalize_job(job),
            'metadata': config.get('metadata', {}),
            'version': config.get('version', '1.0')
        }
        
        return normalized_config
    
    def _validate_persona(self, persona: Dict[str, Any]):
        """Validate persona section."""
        if not isinstance(persona, dict):
            raise PersonaValidationError("Persona section must be a dictionary")
        
        # Check required fields
        for field in self.required_persona_fields:
            if field not in persona:
                raise PersonaValidationError(f"Missing required persona field: {field}")
            if not persona[field]:
                raise PersonaValidationError(f"Empty persona field: {field}")
        
        # Validate specific field types
        if not isinstance(persona['expertise_areas'], list):
            raise PersonaValidationError("expertise_areas must be a list")
        
        if not isinstance(persona['priorities'], list):
            raise PersonaValidationError("priorities must be a list")
        
        if len(persona['expertise_areas']) == 0:
            raise PersonaValidationError("expertise_areas cannot be empty")
    
    def _validate_job(self, job: Dict[str, Any]):
        """Validate job-to-be-done section."""
        if not isinstance(job, dict):
            raise PersonaValidationError("Job-to-be-done section must be a dictionary")
        
        # Check required fields
        for field in self.required_job_fields:
            if field not in job:
                raise PersonaValidationError(f"Missing required job field: {field}")
            if not job[field]:
                raise PersonaValidationError(f"Empty job field: {field}")
        
        # Validate specific field types
        if not isinstance(job['key_objectives'], list):
            raise PersonaValidationError("key_objectives must be a list")
        
        if not isinstance(job['success_criteria'], list):
            raise PersonaValidationError("success_criteria must be a list")
    
    def _normalize_persona(self, persona: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize and enrich persona data."""
        normalized = {
            'role': persona['role'].strip(),
            'description': persona['description'].strip(),
            'expertise_areas': [area.strip() for area in persona['expertise_areas']],
            'priorities': [priority.strip() for priority in persona['priorities']],
            'experience_level': persona.get('experience_level', 'experienced'),
            'industry': persona.get('industry', 'general'),
            'key_skills': persona.get('key_skills', []),
            'preferences': persona.get('preferences', {}),
            'context': persona.get('context', ''),
            'communication_style': persona.get('communication_style', 'professional')
        }
        
        # Create combined description for embedding
        combined_desc_parts = [
            f"Role: {normalized['role']}",
            f"Description: {normalized['description']}",
            f"Expertise: {', '.join(normalized['expertise_areas'])}",
            f"Priorities: {', '.join(normalized['priorities'])}"
        ]
        
        if normalized['context']:
            combined_desc_parts.append(f"Context: {normalized['context']}")
        
        normalized['combined_description'] = ' | '.join(combined_desc_parts)
        
        return normalized
    
    def _normalize_job(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize and enrich job-to-be-done data."""
        normalized = {
            'title': job['title'].strip(),
            'description': job['description'].strip(),
            'key_objectives': [obj.strip() for obj in job['key_objectives']],
            'success_criteria': [criteria.strip() for criteria in job['success_criteria']],
            'urgency': job.get('urgency', 'medium'),
            'scope': job.get('scope', 'focused'),
            'constraints': job.get('constraints', []),
            'stakeholders': job.get('stakeholders', []),
            'timeline': job.get('timeline', ''),
            'expected_outcomes': job.get('expected_outcomes', [])
        }
        
        # Create combined description for embedding
        combined_desc_parts = [
            f"Job: {normalized['title']}",
            f"Description: {normalized['description']}",
            f"Objectives: {', '.join(normalized['key_objectives'])}",
            f"Success criteria: {', '.join(normalized['success_criteria'])}"
        ]
        
        if normalized['constraints']:
            combined_desc_parts.append(f"Constraints: {', '.join(normalized['constraints'])}")
        
        normalized['combined_description'] = ' | '.join(combined_desc_parts)
        
        return normalized
    
    def get_default_persona(self) -> Dict[str, Any]:
        """Get a default persona configuration for fallback."""
        return {
            'persona': {
                'role': 'Business Analyst',
                'description': 'A professional who analyzes business processes, requirements, and documents to provide insights and recommendations.',
                'expertise_areas': [
                    'Business process analysis',
                    'Requirements gathering',
                    'Data analysis',
                    'Documentation review',
                    'Strategic planning'
                ],
                'priorities': [
                    'Accuracy and attention to detail',
                    'Clear communication',
                    'Actionable insights',
                    'Efficiency improvements'
                ],
                'experience_level': 'experienced',
                'industry': 'general',
                'key_skills': [
                    'Critical thinking',
                    'Problem solving',
                    'Communication',
                    'Data interpretation'
                ],
                'preferences': {
                    'detail_level': 'comprehensive',
                    'format_preference': 'structured',
                    'focus_areas': ['key_findings', 'recommendations', 'action_items']
                },
                'context': 'Analyzing documents to extract relevant information for business decision making',
                'communication_style': 'professional',
                'combined_description': 'Role: Business Analyst | Description: A professional who analyzes business processes, requirements, and documents to provide insights and recommendations. | Expertise: Business process analysis, Requirements gathering, Data analysis, Documentation review, Strategic planning | Priorities: Accuracy and attention to detail, Clear communication, Actionable insights, Efficiency improvements | Context: Analyzing documents to extract relevant information for business decision making'
            },
            'job_to_be_done': {
                'title': 'Extract Relevant Information from Documents',
                'description': 'Identify and extract the most relevant sections and information from documents that align with the persona\'s expertise and current needs.',
                'key_objectives': [
                    'Find information relevant to the persona\'s role',
                    'Prioritize content based on persona\'s expertise areas',
                    'Extract actionable insights',
                    'Identify key findings and recommendations'
                ],
                'success_criteria': [
                    'High relevance score for extracted content',
                    'Coverage of persona\'s key interest areas',
                    'Actionable and specific information',
                    'Well-structured and organized output'
                ],
                'urgency': 'medium',
                'scope': 'focused',
                'constraints': [
                    'Must maintain accuracy',
                    'Should avoid information overload',
                    'Focus on persona\'s expertise areas'
                ],
                'stakeholders': ['Document analyst', 'Business stakeholders'],
                'timeline': 'Immediate',
                'expected_outcomes': [
                    'Ranked list of relevant document sections',
                    'Key insights aligned with persona needs',
                    'Structured output for easy consumption'
                ],
                'combined_description': 'Job: Extract Relevant Information from Documents | Description: Identify and extract the most relevant sections and information from documents that align with the persona\'s expertise and current needs. | Objectives: Find information relevant to the persona\'s role, Prioritize content based on persona\'s expertise areas, Extract actionable insights, Identify key findings and recommendations | Success criteria: High relevance score for extracted content, Coverage of persona\'s key interest areas, Actionable and specific information, Well-structured and organized output | Constraints: Must maintain accuracy, Should avoid information overload, Focus on persona\'s expertise areas'
            },
            'metadata': {
                'created_by': 'system',
                'created_date': '2024-01-01',
                'description': 'Default persona configuration for general business analysis tasks'
            },
            'version': '1.0'
        }
    
    def create_persona_template(self, output_path: str, format_type: str = 'json'):
        """
        Create a template file for persona configuration.
        
        Args:
            output_path: Path where template will be saved
            format_type: 'json' or 'yaml'
        """
        template = {
            'persona': {
                'role': 'Your Role Title',
                'description': 'Detailed description of the persona\'s background, experience, and responsibilities',
                'expertise_areas': [
                    'Area of expertise 1',
                    'Area of expertise 2',
                    'Area of expertise 3'
                ],
                'priorities': [
                    'What this persona values most',
                    'Secondary priority',
                    'Additional consideration'
                ],
                'experience_level': 'junior|experienced|senior|expert',
                'industry': 'specific industry or general',
                'key_skills': [
                    'Skill 1',
                    'Skill 2'
                ],
                'preferences': {
                    'detail_level': 'summary|detailed|comprehensive',
                    'format_preference': 'bullet_points|structured|narrative',
                    'focus_areas': ['area1', 'area2']
                },
                'context': 'Additional context about the persona\'s current situation or needs',
                'communication_style': 'professional|casual|technical|friendly'
            },
            'job_to_be_done': {
                'title': 'What job is this persona trying to accomplish?',
                'description': 'Detailed description of the task or objective',
                'key_objectives': [
                    'Primary objective',
                    'Secondary objective',
                    'Additional goal'
                ],
                'success_criteria': [
                    'How will success be measured?',
                    'What constitutes a good outcome?'
                ],
                'urgency': 'low|medium|high|critical',
                'scope': 'narrow|focused|broad|comprehensive',
                'constraints': [
                    'Limitation or constraint 1',
                    'Limitation or constraint 2'
                ],
                'stakeholders': [
                    'Who else is involved or affected?'
                ],
                'timeline': 'When does this need to be completed?',
                'expected_outcomes': [
                    'Expected result 1',
                    'Expected result 2'
                ]
            },
            'metadata': {
                'created_by': 'Your name',
                'created_date': '2024-01-01',
                'description': 'Purpose of this persona configuration',
                'version': '1.0'
            }
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            if format_type.lower() == 'json':
                json.dump(template, f, indent=2, ensure_ascii=False)
            else:  # YAML
                yaml.dump(template, f, default_flow_style=False, allow_unicode=True, indent=2)
        
        logger.info(f"Created persona template: {output_path}")
    
    def validate_file(self, file_path: str) -> Dict[str, Any]:
        """
        Validate a persona configuration file without loading it fully.
        
        Returns:
            Dictionary with validation results
        """
        try:
            config = self.load_from_file(file_path)
            return {
                'valid': True,
                'persona_role': config['persona']['role'],
                'job_title': config['job_to_be_done']['title'],
                'version': config.get('version', 'unknown'),
                'message': 'Configuration is valid'
            }
        except Exception as e:
            return {
                'valid': False,
                'error': str(e),
                'message': 'Configuration validation failed'
            }
    
    def get_persona_summary(self, config: Dict[str, Any]) -> str:
        """Get a human-readable summary of the persona configuration."""
        persona = config['persona']
        job = config['job_to_be_done']
        
        summary_parts = [
            f"👤 Persona: {persona['role']}",
            f"📋 Job: {job['title']}",
            f"🎯 Expertise: {', '.join(persona['expertise_areas'][:3])}{'...' if len(persona['expertise_areas']) > 3 else ''}",
            f"⚡ Priorities: {', '.join(persona['priorities'][:2])}{'...' if len(persona['priorities']) > 2 else ''}",
            f"🏆 Success: {', '.join(job['success_criteria'][:2])}{'...' if len(job['success_criteria']) > 2 else ''}"
        ]
        
        return '\n'.join(summary_parts)


# Utility functions
def load_persona_quick(file_path: str) -> Dict[str, str]:
    """Quick load of persona for basic information."""
    parser = PersonaParser()
    try:
        config = parser.load_from_file(file_path)
        return {
            'role': config['persona']['role'],
            'job': config['job_to_be_done']['title'],
            'persona_desc': config['persona']['combined_description'],
            'job_desc': config['job_to_be_done']['combined_description']
        }
    except Exception as e:
        logger.error(f"Quick persona load failed: {str(e)}")
        return parser.get_default_persona()


def create_persona_from_text(role: str, description: str, job_title: str, job_description: str) -> Dict[str, Any]:
    """Create a basic persona configuration from text inputs."""
    return {
        'persona': {
            'role': role,
            'description': description,
            'expertise_areas': [role.lower().replace(' ', '_')],
            'priorities': ['accuracy', 'efficiency'],
            'combined_description': f"Role: {role} | Description: {description}"
        },
        'job_to_be_done': {
            'title': job_title,
            'description': job_description,
            'key_objectives': ['complete_task'],
            'success_criteria': ['successful_completion'],
            'combined_description': f"Job: {job_title} | Description: {job_description}"
        },
        'metadata': {'created_by': 'text_input', 'version': '1.0'}
    }