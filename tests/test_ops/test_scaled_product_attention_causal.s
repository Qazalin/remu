
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <r_16_16>:
	s_add_i32 s3, s15, -1                                      // 000000001600: 8103C10F
	s_load_b64 s[0:1], s[0:1], null                            // 000000001604: F4040000 F8000000
	s_cmp_gt_i32 s3, 13                                        // 00000000160C: BF028D03
	s_mov_b32 s2, s15                                          // 000000001610: BE82000F
	s_cselect_b32 s3, -1, 0                                    // 000000001614: 980380C1
	s_cmp_gt_i32 s15, 13                                       // 000000001618: BF028D0F
	v_cndmask_b32_e64 v0, 0, 1, s3                             // 00000000161C: D5010000 000D0280
	s_cselect_b32 s3, -1, 0                                    // 000000001624: 980380C1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001628: BF8704A9
	v_cndmask_b32_e64 v1, 0, 1, s3                             // 00000000162C: D5010001 000D0280
	s_add_i32 s3, s15, 1                                       // 000000001634: 8103810F
	s_cmp_gt_i32 s3, 13                                        // 000000001638: BF028D03
	s_cselect_b32 vcc_lo, -1, 0                                // 00000000163C: 986A80C1
	s_add_i32 s3, s15, 2                                       // 000000001640: 8103820F
	v_add_co_ci_u32_e32 v0, vcc_lo, v0, v1, vcc_lo             // 000000001644: 40000300
	s_cmp_gt_i32 s3, 13                                        // 000000001648: BF028D03
	s_cselect_b32 s3, -1, 0                                    // 00000000164C: 980380C1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001650: BF8704A9
	v_cndmask_b32_e64 v1, 0, 1, s3                             // 000000001654: D5010001 000D0280
	s_add_i32 s3, s15, 3                                       // 00000000165C: 8103830F
	s_cmp_gt_i32 s3, 13                                        // 000000001660: BF028D03
	s_cselect_b32 vcc_lo, -1, 0                                // 000000001664: 986A80C1
	s_add_i32 s3, s15, 4                                       // 000000001668: 8103840F
	v_add_co_ci_u32_e32 v0, vcc_lo, v0, v1, vcc_lo             // 00000000166C: 40000300
	s_cmp_gt_i32 s3, 13                                        // 000000001670: BF028D03
	s_cselect_b32 s3, -1, 0                                    // 000000001674: 980380C1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001678: BF8704A9
	v_cndmask_b32_e64 v1, 0, 1, s3                             // 00000000167C: D5010001 000D0280
	s_add_i32 s3, s15, 5                                       // 000000001684: 8103850F
	s_cmp_gt_i32 s3, 13                                        // 000000001688: BF028D03
	s_cselect_b32 vcc_lo, -1, 0                                // 00000000168C: 986A80C1
	s_add_i32 s3, s15, 6                                       // 000000001690: 8103860F
	v_add_co_ci_u32_e32 v0, vcc_lo, v0, v1, vcc_lo             // 000000001694: 40000300
	s_cmp_gt_i32 s3, 13                                        // 000000001698: BF028D03
	s_cselect_b32 s3, -1, 0                                    // 00000000169C: 980380C1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 0000000016A0: BF8704A9
	v_cndmask_b32_e64 v1, 0, 1, s3                             // 0000000016A4: D5010001 000D0280
	s_add_i32 s3, s15, 7                                       // 0000000016AC: 8103870F
	s_cmp_gt_i32 s3, 13                                        // 0000000016B0: BF028D03
	s_cselect_b32 vcc_lo, -1, 0                                // 0000000016B4: 986A80C1
	s_add_i32 s3, s15, 8                                       // 0000000016B8: 8103880F
	v_add_co_ci_u32_e32 v0, vcc_lo, v0, v1, vcc_lo             // 0000000016BC: 40000300
	s_cmp_gt_i32 s3, 13                                        // 0000000016C0: BF028D03
	s_cselect_b32 s3, -1, 0                                    // 0000000016C4: 980380C1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 0000000016C8: BF8704A9
	v_cndmask_b32_e64 v1, 0, 1, s3                             // 0000000016CC: D5010001 000D0280
	s_add_i32 s3, s15, 9                                       // 0000000016D4: 8103890F
	s_cmp_gt_i32 s3, 13                                        // 0000000016D8: BF028D03
	s_cselect_b32 vcc_lo, -1, 0                                // 0000000016DC: 986A80C1
	s_add_i32 s3, s15, 10                                      // 0000000016E0: 81038A0F
	v_add_co_ci_u32_e32 v0, vcc_lo, v0, v1, vcc_lo             // 0000000016E4: 40000300
	s_cmp_gt_i32 s3, 13                                        // 0000000016E8: BF028D03
	s_cselect_b32 s3, -1, 0                                    // 0000000016EC: 980380C1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 0000000016F0: BF8704A9
	v_cndmask_b32_e64 v1, 0, 1, s3                             // 0000000016F4: D5010001 000D0280
	s_add_i32 s3, s15, 11                                      // 0000000016FC: 81038B0F
	s_cmp_gt_i32 s3, 13                                        // 000000001700: BF028D03
	s_cselect_b32 vcc_lo, -1, 0                                // 000000001704: 986A80C1
	s_add_i32 s3, s15, 12                                      // 000000001708: 81038C0F
	v_add_co_ci_u32_e32 v0, vcc_lo, v0, v1, vcc_lo             // 00000000170C: 40000300
	s_cmp_gt_i32 s3, 13                                        // 000000001710: BF028D03
	s_cselect_b32 s3, -1, 0                                    // 000000001714: 980380C1
	s_add_i32 s4, s15, 13                                      // 000000001718: 81048D0F
	v_cndmask_b32_e64 v1, 0, 1, s3                             // 00000000171C: D5010001 000D0280
	s_cmp_gt_i32 s4, 13                                        // 000000001724: BF028D04
	s_cselect_b32 vcc_lo, -1, 0                                // 000000001728: 986A80C1
	s_cmp_lt_u32 s15, 0x7ffffff2                               // 00000000172C: BF0AFF0F 7FFFFFF2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_2)// 000000001734: BF870141
	v_add_co_ci_u32_e32 v0, vcc_lo, v0, v1, vcc_lo             // 000000001738: 40000300
	s_cselect_b32 vcc_lo, -1, 0                                // 00000000173C: 986A80C1
	s_ashr_i32 s3, s15, 31                                     // 000000001740: 86039F0F
	v_mov_b32_e32 v1, 0                                        // 000000001744: 7E020280
	v_add_co_ci_u32_e32 v0, vcc_lo, -2, v0, vcc_lo             // 000000001748: 400000C2
	s_lshl_b64 s[2:3], s[2:3], 2                               // 00000000174C: 84828202
	s_waitcnt lgkmcnt(0)                                       // 000000001750: BF89FC07
	s_add_u32 s0, s0, s2                                       // 000000001754: 80000200
	s_addc_u32 s1, s1, s3                                      // 000000001758: 82010301
	global_store_b32 v1, v0, s[0:1]                            // 00000000175C: DC6A0000 00000001
	s_nop 0                                                    // 000000001764: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001768: BFB60003
	s_endpgm                                                   // 00000000176C: BFB00000
