# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from fastapi import APIRouter

router = APIRouter()

@router.get("/health", tags=["Health API"], summary="Check API health")
@router.get("/health/", include_in_schema=False)
async def health():
    """

    **Response**:

    - **status** (string): A string describing health status.
    """
    return {"status": "healthy"}