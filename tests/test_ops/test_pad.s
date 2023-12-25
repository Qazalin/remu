
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <E_10_6>:
	s_load_b128 s[4:7], s[0:1], null                           // 000000001700: F4080100 F8000000
	s_add_i32 s0, s14, -1                                      // 000000001708: 8100C10E
	s_mov_b32 s2, s15                                          // 00000000170C: BE82000F
	s_cmp_gt_u32 s0, 2                                         // 000000001710: BF088200
	s_cselect_b32 s0, -1, 0                                    // 000000001714: 980080C1
	s_add_i32 s1, s15, -3                                      // 000000001718: 8101C30F
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 00000000171C: BF8704A9
	s_cmp_gt_u32 s1, 2                                         // 000000001720: BF088201
	s_cselect_b32 s1, -1, 0                                    // 000000001724: 980180C1
	s_or_b32 s0, s0, s1                                        // 000000001728: 8C000100
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000172C: BF870009
	s_and_b32 vcc_lo, exec_lo, s0                              // 000000001730: 8B6A007E
	s_mov_b32 s0, 0                                            // 000000001734: BE800080
	s_cbranch_vccnz 14                                         // 000000001738: BFA4000E <E_10_6+0x74>
	s_mul_i32 s0, s2, 3                                        // 00000000173C: 96008302
	s_mov_b32 s1, 0                                            // 000000001740: BE810080
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001744: BF870009
	s_lshl_b64 s[8:9], s[0:1], 2                               // 000000001748: 84888200
	s_mov_b32 s15, s1                                          // 00000000174C: BE8F0001
	s_waitcnt lgkmcnt(0)                                       // 000000001750: BF89FC07
	s_add_u32 s3, s6, s8                                       // 000000001754: 80030806
	s_addc_u32 s6, s7, s9                                      // 000000001758: 82060907
	s_lshl_b64 s[0:1], s[14:15], 2                             // 00000000175C: 8480820E
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001760: BF870009
	s_add_u32 s0, s3, s0                                       // 000000001764: 80000003
	s_addc_u32 s1, s6, s1                                      // 000000001768: 82010106
	s_load_b32 s0, s[0:1], -0x28                               // 00000000176C: F4000000 F81FFFD8
	s_mul_i32 s2, s2, 6                                        // 000000001774: 96028602
	s_waitcnt lgkmcnt(0)                                       // 000000001778: BF89FC07
	v_dual_mov_b32 v0, 0 :: v_dual_mov_b32 v1, s0              // 00000000177C: CA100080 00000000
	s_ashr_i32 s3, s2, 31                                      // 000000001784: 86039F02
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001788: BF870499
	s_lshl_b64 s[2:3], s[2:3], 2                               // 00000000178C: 84828202
	s_add_u32 s1, s4, s2                                       // 000000001790: 80010204
	s_addc_u32 s4, s5, s3                                      // 000000001794: 82040305
	s_ashr_i32 s15, s14, 31                                    // 000000001798: 860F9F0E
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000179C: BF870499
	s_lshl_b64 s[2:3], s[14:15], 2                             // 0000000017A0: 8482820E
	s_add_u32 s0, s1, s2                                       // 0000000017A4: 80000201
	s_addc_u32 s1, s4, s3                                      // 0000000017A8: 82010304
	global_store_b32 v0, v1, s[0:1]                            // 0000000017AC: DC6A0000 00000100
	s_nop 0                                                    // 0000000017B4: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000017B8: BFB60003
	s_endpgm                                                   // 0000000017BC: BFB00000
