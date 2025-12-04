from mappers.cpt_snomed_mapper import CptSnomedMapper
from mappers.mapper import Mapper
from mappers.mapper_enum import MapperEnum
from mappers.ndc_snomed_mapper import NdcSnomedMapper

mapper: dict[MapperEnum, Mapper] = {
    MapperEnum.CPT_SNOMED: CptSnomedMapper(),
    MapperEnum.NDC_SNOMED: NdcSnomedMapper(),
}

__all__ = ["mapper"]
