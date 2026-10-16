import { Router, type IRouter } from "express";
import confidenceCollapseRouter from "./confidence-collapse";
import healthRouter from "./health";

const router: IRouter = Router();

router.use(healthRouter);
router.use(confidenceCollapseRouter);

export default router;
