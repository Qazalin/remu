
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <r_256_256>:
	v_mov_b32_e32 v0, 0                                        // 000000001600: 7E000280
	s_mov_b32 s2, s15                                          // 000000001604: BE82000F
	s_mov_b32 s3, 0                                            // 000000001608: BE830080
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000160C: BF870499
	s_add_i32 s4, s2, s3                                       // 000000001610: 81040302
	s_add_i32 s5, s4, -1                                       // 000000001614: 8105C104
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001618: BF870009
	s_cmpk_gt_i32 s5, 0xfd                                     // 00000000161C: B28500FD
	s_cselect_b32 s5, -1, 0                                    // 000000001620: 980580C1
	s_cmpk_gt_i32 s4, 0xfd                                     // 000000001624: B28400FD
	v_cndmask_b32_e64 v1, 0, 1, s5                             // 000000001628: D5010001 00150280
	s_cselect_b32 vcc_lo, -1, 0                                // 000000001630: 986A80C1
	s_add_i32 s5, s4, 1                                        // 000000001634: 81058104
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001638: BF870099
	s_cmpk_gt_i32 s5, 0xfd                                     // 00000000163C: B28500FD
	v_add_co_ci_u32_e32 v0, vcc_lo, v0, v1, vcc_lo             // 000000001640: 40000300
	s_cselect_b32 s5, -1, 0                                    // 000000001644: 980580C1
	s_add_i32 s6, s4, 2                                        // 000000001648: 81068204
	v_cndmask_b32_e64 v1, 0, 1, s5                             // 00000000164C: D5010001 00150280
	s_cmpk_gt_i32 s6, 0xfd                                     // 000000001654: B28600FD
	s_cselect_b32 vcc_lo, -1, 0                                // 000000001658: 986A80C1
	s_add_i32 s5, s4, 3                                        // 00000000165C: 81058304
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001660: BF870001
	v_add_co_ci_u32_e32 v0, vcc_lo, v0, v1, vcc_lo             // 000000001664: 40000300
	s_cmpk_gt_i32 s5, 0xfd                                     // 000000001668: B28500FD
	s_cselect_b32 s5, -1, 0                                    // 00000000166C: 980580C1
	s_add_i32 s6, s4, 4                                        // 000000001670: 81068404
	v_cndmask_b32_e64 v1, 0, 1, s5                             // 000000001674: D5010001 00150280
	s_cmpk_gt_i32 s6, 0xfd                                     // 00000000167C: B28600FD
	s_cselect_b32 vcc_lo, -1, 0                                // 000000001680: 986A80C1
	s_add_i32 s5, s4, 5                                        // 000000001684: 81058504
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001688: BF870001
	v_add_co_ci_u32_e32 v0, vcc_lo, v0, v1, vcc_lo             // 00000000168C: 40000300
	s_cmpk_gt_i32 s5, 0xfd                                     // 000000001690: B28500FD
	s_cselect_b32 s5, -1, 0                                    // 000000001694: 980580C1
	s_add_i32 s6, s4, 6                                        // 000000001698: 81068604
	v_cndmask_b32_e64 v1, 0, 1, s5                             // 00000000169C: D5010001 00150280
	s_cmpk_gt_i32 s6, 0xfd                                     // 0000000016A4: B28600FD
	s_cselect_b32 vcc_lo, -1, 0                                // 0000000016A8: 986A80C1
	s_add_i32 s5, s4, 7                                        // 0000000016AC: 81058704
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000016B0: BF870001
	v_add_co_ci_u32_e32 v0, vcc_lo, v0, v1, vcc_lo             // 0000000016B4: 40000300
	s_cmpk_gt_i32 s5, 0xfd                                     // 0000000016B8: B28500FD
	s_cselect_b32 s5, -1, 0                                    // 0000000016BC: 980580C1
	s_add_i32 s6, s4, 8                                        // 0000000016C0: 81068804
	v_cndmask_b32_e64 v1, 0, 1, s5                             // 0000000016C4: D5010001 00150280
	s_cmpk_gt_i32 s6, 0xfd                                     // 0000000016CC: B28600FD
	s_cselect_b32 vcc_lo, -1, 0                                // 0000000016D0: 986A80C1
	s_add_i32 s5, s4, 9                                        // 0000000016D4: 81058904
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000016D8: BF870001
	v_add_co_ci_u32_e32 v0, vcc_lo, v0, v1, vcc_lo             // 0000000016DC: 40000300
	s_cmpk_gt_i32 s5, 0xfd                                     // 0000000016E0: B28500FD
	s_cselect_b32 s5, -1, 0                                    // 0000000016E4: 980580C1
	s_add_i32 s6, s4, 10                                       // 0000000016E8: 81068A04
	v_cndmask_b32_e64 v1, 0, 1, s5                             // 0000000016EC: D5010001 00150280
	s_cmpk_gt_i32 s6, 0xfd                                     // 0000000016F4: B28600FD
	s_cselect_b32 vcc_lo, -1, 0                                // 0000000016F8: 986A80C1
	s_add_i32 s5, s4, 11                                       // 0000000016FC: 81058B04
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001700: BF870001
	v_add_co_ci_u32_e32 v0, vcc_lo, v0, v1, vcc_lo             // 000000001704: 40000300
	s_cmpk_gt_i32 s5, 0xfd                                     // 000000001708: B28500FD
	s_cselect_b32 s5, -1, 0                                    // 00000000170C: 980580C1
	s_add_i32 s6, s4, 12                                       // 000000001710: 81068C04
	v_cndmask_b32_e64 v1, 0, 1, s5                             // 000000001714: D5010001 00150280
	s_cmpk_gt_i32 s6, 0xfd                                     // 00000000171C: B28600FD
	s_cselect_b32 vcc_lo, -1, 0                                // 000000001720: 986A80C1
	s_add_i32 s5, s4, 13                                       // 000000001724: 81058D04
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001728: BF870001
	v_add_co_ci_u32_e32 v0, vcc_lo, v0, v1, vcc_lo             // 00000000172C: 40000300
	s_cmpk_gt_i32 s5, 0xfd                                     // 000000001730: B28500FD
	s_cselect_b32 s5, -1, 0                                    // 000000001734: 980580C1
	s_add_i32 s4, s4, 14                                       // 000000001738: 81048E04
	v_cndmask_b32_e64 v1, 0, 1, s5                             // 00000000173C: D5010001 00150280
	s_cmpk_gt_i32 s4, 0xfd                                     // 000000001744: B28400FD
	s_cselect_b32 vcc_lo, -1, 0                                // 000000001748: 986A80C1
	s_add_i32 s3, s3, 16                                       // 00000000174C: 81039003
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001750: BF870001
	v_add_co_ci_u32_e32 v0, vcc_lo, v0, v1, vcc_lo             // 000000001754: 40000300
	s_cmpk_eq_i32 s3, 0x100                                    // 000000001758: B1830100
	s_cbranch_scc0 65451                                       // 00000000175C: BFA1FFAB <r_256_256+0xc>
	s_load_b64 s[0:1], s[0:1], null                            // 000000001760: F4040000 F8000000
	s_ashr_i32 s3, s2, 31                                      // 000000001768: 86039F02
	v_dual_mov_b32 v1, 0 :: v_dual_add_nc_u32 v0, -1, v0       // 00000000176C: CA200080 010000C1
	s_lshl_b64 s[2:3], s[2:3], 2                               // 000000001774: 84828202
	s_waitcnt lgkmcnt(0)                                       // 000000001778: BF89FC07
	s_add_u32 s0, s0, s2                                       // 00000000177C: 80000200
	s_addc_u32 s1, s1, s3                                      // 000000001780: 82010301
	global_store_b32 v1, v0, s[0:1]                            // 000000001784: DC6A0000 00000001
	s_nop 0                                                    // 00000000178C: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001790: BFB60003
	s_endpgm                                                   // 000000001794: BFB00000
