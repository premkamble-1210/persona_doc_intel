"""
Document ranking and relevance scoring utilities.
Ranks document chunks based on persona and job-to-be-done relevance.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
import re

logger = logging.getLogger(__name__)


@dataclass
class RankingScore:
    """Data class for ranking scores."""
    persona_score: float
    job_score: float
    combined_score: float
    content_quality_score: float
    final_score: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DocumentRanker:
    """
    Advanced document ranking system based on persona and job relevance.
    Uses multiple scoring strategies for robust ranking.
    """
    
    def __init__(self):
        self.scoring_strategies = {
            'cosine_similarity': self._cosine_similarity_score,
            'semantic_overlap': self._semantic_overlap_score,
            'keyword_matching': self._keyword_matching_score,
            'content_quality': self._content_quality_score
        }
        
        # Default weights for different scoring components
        self.default_weights = {
            'persona_weight': 0.6,
            'job_weight': 0.4,
            'content_quality_weight': 0.2,
            'length_penalty_weight': 0.1
        }
    
    def rank_chunks(self, chunks_with_embeddings: List[Dict[str, Any]], 
                   persona_embedding: np.ndarray, job_embedding: np.ndarray,
                   persona_config: Dict[str, Any], top_k: int = 20,
                   custom_weights: Dict[str, float] = None) -> List[Dict[str, Any]]:
        """
        Rank document chunks by relevance to persona and job.
        
        Args:
            chunks_with_embeddings: List of chunks with their embeddings
            persona_embedding: Embedding vector for the persona
            job_embedding: Embedding vector for the job-to-be-done
            persona_config: Full persona configuration
            top_k: Number of top chunks to return
            custom_weights: Custom weights for scoring components
            
        Returns:
            List of ranked chunks with scores
        """
        if not chunks_with_embeddings:
            return []
        
        weights = {**self.default_weights, **(custom_weights or {})}
        
        logger.info(f"Ranking {len(chunks_with_embeddings)} chunks")
        
        ranked_chunks = []
        
        for i, chunk in enumerate(chunks_with_embeddings):
            try:
                # Calculate multiple types of scores
                scores = self._calculate_comprehensive_score(
                    chunk, persona_embedding, job_embedding, persona_config, weights
                )
                
                # Create ranked chunk with all scoring information
                ranked_chunk = {
                    **chunk,
                    'relevance_score': scores.final_score,
                    'persona_score': scores.persona_score,
                    'job_score': scores.job_score,
                    'content_quality_score': scores.content_quality_score,
                    'ranking_metadata': scores.metadata,
                    'rank_position': 0  # Will be set after sorting
                }
                
                # Remove embedding from output to save space (optional)
                if 'embedding' in ranked_chunk:
                    del ranked_chunk['embedding']
                
                ranked_chunks.append(ranked_chunk)
                
            except Exception as e:
                logger.warning(f"Failed to rank chunk {i}: {str(e)}")
                continue
        
        # Sort by final score
        ranked_chunks.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Add rank positions
        for i, chunk in enumerate(ranked_chunks):
            chunk['rank_position'] = i + 1
        
        # Return top K results
        top_chunks = ranked_chunks[:top_k]
        
        logger.info(f"Ranked chunks, top score: {top_chunks[0]['relevance_score']:.3f}" if top_chunks else "No chunks ranked")
        
        return top_chunks
    
    def _calculate_comprehensive_score(self, chunk: Dict[str, Any], 
                                     persona_embedding: np.ndarray, 
                                     job_embedding: np.ndarray,
                                     persona_config: Dict[str, Any],
                                     weights: Dict[str, float]) -> RankingScore:
        """Calculate comprehensive ranking score for a chunk."""
        
        chunk_embedding = np.array(chunk.get('embedding', []))
        chunk_content = chunk.get('content', '')
        
        # 1. Persona relevance score
        persona_score = self._calculate_persona_relevance(
            chunk_embedding, chunk_content, persona_embedding, persona_config
        )
        
        # 2. Job relevance score
        job_score = self._calculate_job_relevance(
            chunk_embedding, chunk_content, job_embedding, persona_config
        )
        
        # 3. Content quality score
        content_quality_score = self._calculate_content_quality(chunk_content)
        
        # 4. Context and position bonuses
        context_bonus = self._calculate_context_bonus(chunk, persona_config)
        
        # 5. Combine scores
        combined_semantic_score = (
            persona_score * weights['persona_weight'] + 
            job_score * weights['job_weight']
        )
        
        # 6. Apply content quality and context adjustments
        final_score = (
            combined_semantic_score * 0.7 + 
            content_quality_score * weights['content_quality_weight'] +
            context_bonus * 0.1
        )
        
        # 7. Length penalty for very short or very long chunks
        length_penalty = self._calculate_length_penalty(chunk_content)
        final_score = final_score * (1.0 - length_penalty * weights['length_penalty_weight'])
        
        # Create detailed metadata
        metadata = {
            'semantic_score': combined_semantic_score,
            'context_bonus': context_bonus,
            'length_penalty': length_penalty,
            'content_length': len(chunk_content),
            'has_keywords': self._has_persona_keywords(chunk_content, persona_config)
        }
        
        return RankingScore(
            persona_score=persona_score,
            job_score=job_score,
            combined_score=combined_semantic_score,
            content_quality_score=content_quality_score,
            final_score=final_score,
            metadata=metadata
        )
    
    def _calculate_persona_relevance(self, chunk_embedding: np.ndarray, 
                                   chunk_content: str, persona_embedding: np.ndarray,
                                   persona_config: Dict[str, Any]) -> float:
        """Calculate how relevant the chunk is to the persona."""
        scores = []
        
        # 1. Semantic similarity with persona description
        if len(chunk_embedding) > 0 and len(persona_embedding) > 0:
            semantic_score = self._cosine_similarity_score(chunk_embedding, persona_embedding)
            scores.append(semantic_score * 0.4)
        
        # 2. Expertise area matching
        expertise_score = self._calculate_expertise_matching(chunk_content, persona_config)
        scores.append(expertise_score * 0.3)
        
        # 3. Priority alignment
        priority_score = self._calculate_priority_alignment(chunk_content, persona_config)
        scores.append(priority_score * 0.2)
        
        # 4. Role-specific keyword matching
        role_score = self._calculate_role_relevance(chunk_content, persona_config)
        scores.append(role_score * 0.1)
        
        return sum(scores) if scores else 0.0
    
    def _calculate_job_relevance(self, chunk_embedding: np.ndarray, 
                               chunk_content: str, job_embedding: np.ndarray,
                               persona_config: Dict[str, Any]) -> float:
        """Calculate how relevant the chunk is to the job-to-be-done."""
        scores = []
        
        # 1. Semantic similarity with job description
        if len(chunk_embedding) > 0 and len(job_embedding) > 0:
            semantic_score = self._cosine_similarity_score(chunk_embedding, job_embedding)
            scores.append(semantic_score * 0.4)
        
        # 2. Objective matching
        objective_score = self._calculate_objective_matching(chunk_content, persona_config)
        scores.append(objective_score * 0.3)
        
        # 3. Success criteria alignment
        success_score = self._calculate_success_criteria_matching(chunk_content, persona_config)
        scores.append(success_score * 0.2)
        
        # 4. Urgency and scope considerations
        urgency_score = self._calculate_urgency_relevance(chunk_content, persona_config)
        scores.append(urgency_score * 0.1)
        
        return sum(scores) if scores else 0.0
    
    def _calculate_expertise_matching(self, content: str, persona_config: Dict[str, Any]) -> float:
        """Calculate how well content matches persona's expertise areas."""
        expertise_areas = persona_config.get('persona', {}).get('expertise_areas', [])
        if not expertise_areas:
            return 0.0
        
        content_lower = content.lower()
        matches = 0
        total_weight = 0
        
        for area in expertise_areas:
            area_keywords = self._extract_keywords_from_expertise(area)
            area_weight = len(area_keywords)
            total_weight += area_weight
            
            for keyword in area_keywords:
                if keyword.lower() in content_lower:
                    matches += 1
        
        return matches / max(total_weight, 1)
    
    def _extract_keywords_from_expertise(self, expertise_area: str) -> List[str]:
        """Extract relevant keywords from an expertise area."""
        # Simple keyword extraction
        keywords = []
        
        # Split by common separators
        terms = re.split(r'[,\s&/]+', expertise_area.lower())
        
        for term in terms:
            term = term.strip()
            if len(term) > 2 and term not in ['and', 'or', 'the', 'of', 'in', 'to', 'for']:
                keywords.append(term)
        
        return keywords
    
    def _calculate_priority_alignment(self, content: str, persona_config: Dict[str, Any]) -> float:
        """Calculate alignment with persona priorities."""
        priorities = persona_config.get('persona', {}).get('priorities', [])
        if not priorities:
            return 0.0
        
        content_lower = content.lower()
        priority_score = 0.0
        
        for priority in priorities:
            priority_keywords = self._extract_keywords_from_expertise(priority)
            for keyword in priority_keywords:
                if keyword in content_lower:
                    priority_score += 1.0 / len(priorities)
        
        return min(priority_score, 1.0)
    
    def _calculate_role_relevance(self, content: str, persona_config: Dict[str, Any]) -> float:
        """Calculate relevance to the persona's role."""
        role = persona_config.get('persona', {}).get('role', '')
        if not role:
            return 0.0
        
        role_keywords = self._extract_keywords_from_expertise(role)
        content_lower = content.lower()
        
        matches = sum(1 for keyword in role_keywords if keyword in content_lower)
        return matches / max(len(role_keywords), 1)
    
    def _calculate_objective_matching(self, content: str, persona_config: Dict[str, Any]) -> float:
        """Calculate matching with job objectives."""
        objectives = persona_config.get('job_to_be_done', {}).get('key_objectives', [])
        if not objectives:
            return 0.0
        
        content_lower = content.lower()
        match_score = 0.0
        
        for objective in objectives:
            objective_keywords = self._extract_keywords_from_expertise(objective)
            for keyword in objective_keywords:
                if keyword in content_lower:
                    match_score += 1.0 / len(objectives)
        
        return min(match_score, 1.0)
    
    def _calculate_success_criteria_matching(self, content: str, persona_config: Dict[str, Any]) -> float:
        """Calculate matching with success criteria."""
        criteria = persona_config.get('job_to_be_done', {}).get('success_criteria', [])
        if not criteria:
            return 0.0
        
        content_lower = content.lower()
        criteria_score = 0.0
        
        for criterion in criteria:
            criterion_keywords = self._extract_keywords_from_expertise(criterion)
            for keyword in criterion_keywords:
                if keyword in content_lower:
                    criteria_score += 1.0 / len(criteria)
        
        return min(criteria_score, 1.0)
    
    def _calculate_urgency_relevance(self, content: str, persona_config: Dict[str, Any]) -> float:
        """Calculate relevance based on urgency indicators."""
        urgency = persona_config.get('job_to_be_done', {}).get('urgency', 'medium')
        
        urgency_indicators = {
            'high': ['urgent', 'immediate', 'critical', 'asap', 'priority', 'emergency'],
            'medium': ['important', 'needed', 'required', 'necessary'],
            'low': ['future', 'later', 'eventually', 'consider']
        }
        
        content_lower = content.lower()
        indicators = urgency_indicators.get(urgency, [])
        
        matches = sum(1 for indicator in indicators if indicator in content_lower)
        return matches / max(len(indicators), 1)
    
    def _calculate_content_quality(self, content: str) -> float:
        """Calculate content quality score."""
        if not content:
            return 0.0
        
        quality_score = 0.0
        
        # 1. Length appropriateness (not too short, not too long)
        length_score = self._calculate_length_quality(content)
        quality_score += length_score * 0.3
        
        # 2. Information density
        info_density = self._calculate_information_density(content)
        quality_score += info_density * 0.3
        
        # 3. Structure and readability
        structure_score = self._calculate_structure_quality(content)
        quality_score += structure_score * 0.2
        
        # 4. Completeness indicators
        completeness_score = self._calculate_completeness(content)
        quality_score += completeness_score * 0.2
        
        return min(quality_score, 1.0)
    
    def _calculate_length_quality(self, content: str) -> float:
        """Calculate quality based on content length."""
        length = len(content)
        
        # Optimal range: 200-800 characters
        if 200 <= length <= 800:
            return 1.0
        elif 100 <= length < 200 or 800 < length <= 1200:
            return 0.8
        elif 50 <= length < 100 or 1200 < length <= 1500:
            return 0.6
        else:
            return 0.3
    
    def _calculate_information_density(self, content: str) -> float:
        """Calculate information density of content."""
        words = content.split()
        if not words:
            return 0.0
        
        # Count meaningful words (not stopwords)
        stopwords = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
                    'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
                    'to', 'was', 'will', 'with', 'this', 'these', 'they', 'we', 'you'}
        
        meaningful_words = sum(1 for word in words if word.lower() not in stopwords)
        density = meaningful_words / len(words)
        
        return min(density * 1.5, 1.0)  # Boost density score
    
    def _calculate_structure_quality(self, content: str) -> float:
        """Calculate structural quality of content."""
        quality_indicators = 0
        
        # Has proper sentences
        if '.' in content or '!' in content or '?' in content:
            quality_indicators += 1
        
        # Has varied sentence structure
        sentences = re.split(r'[.!?]+', content)
        if len(sentences) > 1:
            quality_indicators += 1
        
        # Has paragraphs or structure
        if '\n' in content or len(content) > 300:
            quality_indicators += 1
        
        # Has specific terms or numbers (indicates detailed content)
        if re.search(r'\d+', content) or any(len(word) > 6 for word in content.split()):
            quality_indicators += 1
        
        return quality_indicators / 4.0
    
    def _calculate_completeness(self, content: str) -> float:
        """Calculate content completeness indicators."""
        completeness_score = 0.0
        
        # Check for completeness indicators
        complete_indicators = ['complete', 'total', 'comprehensive', 'detailed', 'full']
        incomplete_indicators = ['partial', 'incomplete', 'ongoing', 'draft', 'preliminary']
        
        content_lower = content.lower()
        
        if any(indicator in content_lower for indicator in complete_indicators):
            completeness_score += 0.5
        
        if any(indicator in content_lower for indicator in incomplete_indicators):
            completeness_score -= 0.3
        
        # Check for conclusion or summary indicators
        conclusion_indicators = ['conclusion', 'summary', 'result', 'outcome', 'finding']
        if any(indicator in content_lower for indicator in conclusion_indicators):
            completeness_score += 0.3
        
        return max(0.0, min(completeness_score + 0.5, 1.0))  # Base score of 0.5
    
    def _calculate_context_bonus(self, chunk: Dict[str, Any], persona_config: Dict[str, Any]) -> float:
        """Calculate context-based bonuses."""
        bonus = 0.0
        
    def _calculate_context_bonus(self, chunk: Dict[str, Any], persona_config: Dict[str, Any]) -> float:
        """Calculate context-based bonuses."""
        bonus = 0.0
        
        # Section title relevance
        section_title = chunk.get('section_title', '')
        if section_title:
            title_relevance = self._calculate_title_relevance(section_title, persona_config)
            bonus += title_relevance * 0.3
        
        # Section level bonus (higher level sections often more important)
        section_level = chunk.get('section_level', 1)
        if section_level == 1:
            bonus += 0.2  # Top-level sections get bonus
        elif section_level == 2:
            bonus += 0.1  # Second-level sections get smaller bonus
        
        # Position bonus (earlier content sometimes more important)
        start_char = chunk.get('start_char', 0)
        if start_char < 5000:  # First ~5000 characters
            bonus += 0.1
        
        return min(bonus, 0.5)  # Cap bonus at 0.5
    
    def _calculate_title_relevance(self, title: str, persona_config: Dict[str, Any]) -> float:
        """Calculate relevance of section title to persona and job."""
        if not title:
            return 0.0
        
        title_lower = title.lower()
        
        # Check against persona expertise areas
        expertise_areas = persona_config.get('persona', {}).get('expertise_areas', [])
        expertise_match = 0.0
        for area in expertise_areas:
            area_keywords = self._extract_keywords_from_expertise(area)
            matches = sum(1 for keyword in area_keywords if keyword in title_lower)
            expertise_match += matches / max(len(area_keywords), 1)
        
        expertise_score = min(expertise_match / max(len(expertise_areas), 1), 1.0)
        
        # Check against job objectives
        objectives = persona_config.get('job_to_be_done', {}).get('key_objectives', [])
        objective_match = 0.0
        for objective in objectives:
            obj_keywords = self._extract_keywords_from_expertise(objective)
            matches = sum(1 for keyword in obj_keywords if keyword in title_lower)
            objective_match += matches / max(len(obj_keywords), 1)
        
        objective_score = min(objective_match / max(len(objectives), 1), 1.0)
        
        return (expertise_score * 0.6 + objective_score * 0.4)
    
    def _calculate_length_penalty(self, content: str) -> float:
        """Calculate penalty for inappropriate content length."""
        length = len(content)
        
        if length < 50:  # Too short
            return 0.3
        elif length > 2000:  # Too long
            return 0.2
        elif length < 100:  # Quite short
            return 0.1
        else:
            return 0.0
    
    def _has_persona_keywords(self, content: str, persona_config: Dict[str, Any]) -> bool:
        """Check if content contains persona-relevant keywords."""
        content_lower = content.lower()
        
        # Check expertise areas
        expertise_areas = persona_config.get('persona', {}).get('expertise_areas', [])
        for area in expertise_areas:
            area_keywords = self._extract_keywords_from_expertise(area)
            if any(keyword in content_lower for keyword in area_keywords):
                return True
        
        # Check job objectives
        objectives = persona_config.get('job_to_be_done', {}).get('key_objectives', [])
        for objective in objectives:
            obj_keywords = self._extract_keywords_from_expertise(objective)
            if any(keyword in content_lower for keyword in obj_keywords):
                return True
        
        return False
    
    def _cosine_similarity_score(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        try:
            if len(embedding1) == 0 or len(embedding2) == 0:
                return 0.0
            
            # Reshape to ensure 2D arrays for sklearn
            emb1 = embedding1.reshape(1, -1)
            emb2 = embedding2.reshape(1, -1)
            
            similarity = cosine_similarity(emb1, emb2)[0][0]
            return float(similarity)
            
        except Exception as e:
            logger.warning(f"Cosine similarity calculation failed: {str(e)}")
            return 0.0
    
    def _semantic_overlap_score(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate semantic overlap score (alternative to cosine similarity)."""
        try:
            if len(embedding1) == 0 or len(embedding2) == 0:
                return 0.0
            
            # Normalize embeddings
            norm1 = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
            norm2 = embedding2 / (np.linalg.norm(embedding2) + 1e-8)
            
            # Calculate element-wise product and sum
            overlap = np.sum(norm1 * norm2)
            return float(overlap)
            
        except Exception as e:
            logger.warning(f"Semantic overlap calculation failed: {str(e)}")
            return 0.0
    
    def _keyword_matching_score(self, text1: str, text2: str) -> float:
        """Calculate keyword matching score between two texts."""
        try:
            # Extract keywords (simple approach)
            words1 = set(word.lower() for word in re.findall(r'\b\w+\b', text1) if len(word) > 3)
            words2 = set(word.lower() for word in re.findall(r'\b\w+\b', text2) if len(word) > 3)
            
            if not words1 or not words2:
                return 0.0
            
            # Calculate Jaccard similarity
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"Keyword matching failed: {str(e)}")
            return 0.0
    
    def _content_quality_score(self, content: str) -> float:
        """Calculate standalone content quality score."""
        return self._calculate_content_quality(content)
    
    def get_ranking_explanation(self, ranked_chunk: Dict[str, Any]) -> str:
        """Generate human-readable explanation of ranking."""
        score = ranked_chunk.get('relevance_score', 0)
        persona_score = ranked_chunk.get('persona_score', 0)
        job_score = ranked_chunk.get('job_score', 0)
        quality_score = ranked_chunk.get('content_quality_score', 0)
        metadata = ranked_chunk.get('ranking_metadata', {})
        
        explanation_parts = [
            f"Overall Relevance: {score:.3f}",
            f"Persona Match: {persona_score:.3f}",
            f"Job Match: {job_score:.3f}",
            f"Content Quality: {quality_score:.3f}"
        ]
        
        if metadata.get('has_keywords'):
            explanation_parts.append("✓ Contains relevant keywords")
        
        if metadata.get('context_bonus', 0) > 0:
            explanation_parts.append("✓ Contextual relevance bonus")
        
        if metadata.get('length_penalty', 0) > 0:
            explanation_parts.append("⚠ Length penalty applied")
        
        return " | ".join(explanation_parts)
    
    def analyze_ranking_distribution(self, ranked_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the distribution of ranking scores."""
        if not ranked_chunks:
            return {}
        
        scores = [chunk['relevance_score'] for chunk in ranked_chunks]
        persona_scores = [chunk.get('persona_score', 0) for chunk in ranked_chunks]
        job_scores = [chunk.get('job_score', 0) for chunk in ranked_chunks]
        
        return {
            'total_chunks': len(ranked_chunks),
            'score_stats': {
                'mean': np.mean(scores),
                'median': np.median(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores)
            },
            'persona_score_stats': {
                'mean': np.mean(persona_scores),
                'median': np.median(persona_scores)
            },
            'job_score_stats': {
                'mean': np.mean(job_scores),
                'median': np.median(job_scores)
            },
            'high_quality_chunks': sum(1 for score in scores if score > 0.7),
            'medium_quality_chunks': sum(1 for score in scores if 0.4 <= score <= 0.7),
            'low_quality_chunks': sum(1 for score in scores if score < 0.4)
        }
    
    def filter_by_threshold(self, ranked_chunks: List[Dict[str, Any]], 
                          threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Filter chunks by minimum relevance threshold."""
        return [chunk for chunk in ranked_chunks 
                if chunk.get('relevance_score', 0) >= threshold]
    
    def re_rank_with_diversity(self, ranked_chunks: List[Dict[str, Any]], 
                             diversity_weight: float = 0.3) -> List[Dict[str, Any]]:
        """Re-rank chunks to promote diversity while maintaining relevance."""
        if len(ranked_chunks) <= 1:
            return ranked_chunks
        
        final_ranking = []
        remaining_chunks = ranked_chunks.copy()
        
        # Always take the highest scored chunk first
        final_ranking.append(remaining_chunks.pop(0))
        
        while remaining_chunks:
            best_chunk = None
            best_score = -1
            best_index = -1
            
            for i, chunk in enumerate(remaining_chunks):
                # Calculate diversity penalty
                diversity_penalty = self._calculate_diversity_penalty(
                    chunk, final_ranking
                )
                
                # Combine relevance and diversity
                adjusted_score = (
                    chunk['relevance_score'] * (1 - diversity_weight) - 
                    diversity_penalty * diversity_weight
                )
                
                if adjusted_score > best_score:
                    best_score = adjusted_score
                    best_chunk = chunk
                    best_index = i
            
            if best_chunk:
                final_ranking.append(remaining_chunks.pop(best_index))
            else:
                break
        
        # Update rank positions
        for i, chunk in enumerate(final_ranking):
            chunk['rank_position'] = i + 1
            chunk['diversity_adjusted'] = True
        
        return final_ranking
    
    def _calculate_diversity_penalty(self, chunk: Dict[str, Any], 
                                   selected_chunks: List[Dict[str, Any]]) -> float:
        """Calculate penalty for lack of diversity."""
        if not selected_chunks:
            return 0.0
        
        content = chunk.get('content', '')
        section_title = chunk.get('section_title', '')
        
        penalty = 0.0
        
        for selected in selected_chunks:
            selected_content = selected.get('content', '')
            selected_title = selected.get('section_title', '')
            
            # Penalty for similar content
            content_similarity = self._keyword_matching_score(content, selected_content)
            penalty += content_similarity * 0.5
            
            # Penalty for same section
            if section_title and section_title == selected_title:
                penalty += 0.3
        
        return min(penalty / len(selected_chunks), 1.0)


class RankingEvaluator:
    """Evaluate and optimize ranking performance."""
    
    def __init__(self):
        self.evaluation_metrics = {}
    
    def evaluate_ranking_quality(self, ranked_chunks: List[Dict[str, Any]], 
                                ground_truth: List[str] = None) -> Dict[str, Any]:
        """Evaluate the quality of ranking results."""
        if not ranked_chunks:
            return {'error': 'No chunks to evaluate'}
        
        evaluation = {
            'total_chunks': len(ranked_chunks),
            'score_distribution': self._analyze_score_distribution(ranked_chunks),
            'content_coverage': self._analyze_content_coverage(ranked_chunks),
            'ranking_consistency': self._analyze_ranking_consistency(ranked_chunks)
        }
        
        if ground_truth:
            evaluation['ground_truth_metrics'] = self._compare_with_ground_truth(
                ranked_chunks, ground_truth
            )
        
        return evaluation
    
    def _analyze_score_distribution(self, ranked_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the distribution of scores."""
        scores = [chunk.get('relevance_score', 0) for chunk in ranked_chunks]
        
        return {
            'mean_score': np.mean(scores),
            'score_range': np.max(scores) - np.min(scores),
            'high_scores': sum(1 for s in scores if s > 0.7),
            'medium_scores': sum(1 for s in scores if 0.4 <= s <= 0.7),
            'low_scores': sum(1 for s in scores if s < 0.4)
        }
    
    def _analyze_content_coverage(self, ranked_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze how well the ranking covers different content areas."""
        sections = set()
        total_content_length = 0
        
        for chunk in ranked_chunks:
            if chunk.get('section_title'):
                sections.add(chunk['section_title'])
            total_content_length += len(chunk.get('content', ''))
        
        return {
            'unique_sections': len(sections),
            'total_content_length': total_content_length,
            'avg_content_per_chunk': total_content_length / len(ranked_chunks)
        }
    
    def _analyze_ranking_consistency(self, ranked_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze consistency of ranking scores."""
        scores = [chunk.get('relevance_score', 0) for chunk in ranked_chunks]
        
        # Check for proper descending order
        is_properly_ordered = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
        
        # Calculate score gaps
        score_gaps = [scores[i] - scores[i+1] for i in range(len(scores)-1)]
        
        return {
            'properly_ordered': is_properly_ordered,
            'avg_score_gap': np.mean(score_gaps) if score_gaps else 0,
            'max_score_gap': np.max(score_gaps) if score_gaps else 0,
            'min_score_gap': np.min(score_gaps) if score_gaps else 0
        }
    
    def _compare_with_ground_truth(self, ranked_chunks: List[Dict[str, Any]], 
                                 ground_truth: List[str]) -> Dict[str, Any]:
        """Compare ranking with ground truth relevance."""
        # Simple implementation - can be extended based on ground truth format
        chunk_contents = [chunk.get('content', '') for chunk in ranked_chunks[:len(ground_truth)]]
        
        # Calculate how many ground truth items appear in top results
        matches = sum(1 for gt_item in ground_truth 
                     if any(gt_item.lower() in content.lower() for content in chunk_contents))
        
        return {
            'precision_at_k': matches / len(chunk_contents) if chunk_contents else 0,
            'recall_at_k': matches / len(ground_truth) if ground_truth else 0,
            'matches_found': matches
        }


# Utility functions
def quick_rank(chunks: List[str], persona_desc: str, job_desc: str, 
               top_k: int = 10) -> List[Tuple[str, float]]:
    """Quick ranking function for simple use cases."""
    from .embedding_model import EmbeddingModel
    
    # This is a simplified version for demonstration
    # In practice, you'd use the full DocumentRanker
    ranker = DocumentRanker()
    model = EmbeddingModel()
    
    # Create simple chunk format
    chunks_with_embeddings = []
    for i, chunk in enumerate(chunks):
        embedding = model.encode(chunk)
        chunks_with_embeddings.append({
            'content': chunk,
            'embedding': embedding,
            'chunk_id': f'chunk_{i}'
        })
    
    # Create simple persona config
    persona_config = {
        'persona': {'expertise_areas': [persona_desc], 'priorities': []},
        'job_to_be_done': {'key_objectives': [job_desc], 'success_criteria': []}
    }
    
    persona_embedding = model.encode(persona_desc)
    job_embedding = model.encode(job_desc)
    
    ranked_chunks = ranker.rank_chunks(
        chunks_with_embeddings, persona_embedding, job_embedding, 
        persona_config, top_k
    )
    
    return [(chunk['content'], chunk['relevance_score']) for chunk in ranked_chunks]