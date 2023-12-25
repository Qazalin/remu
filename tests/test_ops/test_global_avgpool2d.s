
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <r_64_3108>:
	s_load_b128 s[16:19], s[0:1], null                         // 000000001600: F4080400 F8000000
	s_mul_i32 s0, s15, 0xc24                                   // 000000001608: 9600FF0F 00000C24
	v_mov_b32_e32 v0, 0                                        // 000000001610: 7E000280
	s_ashr_i32 s1, s0, 31                                      // 000000001614: 86019F00
	s_mov_b32 s20, s15                                         // 000000001618: BE94000F
	s_lshl_b64 s[0:1], s[0:1], 2                               // 00000000161C: 84808200
	s_waitcnt lgkmcnt(0)                                       // 000000001620: BF89FC07
	s_add_u32 s21, s18, s0                                     // 000000001624: 80150012
	s_addc_u32 s22, s19, s1                                    // 000000001628: 82160113
	s_mov_b64 s[18:19], 0                                      // 00000000162C: BE920180
	s_set_inst_prefetch_distance 0x1                           // 000000001630: BF840001
	s_nop 0                                                    // 000000001634: BF800000
	s_nop 0                                                    // 000000001638: BF800000
	s_nop 0                                                    // 00000000163C: BF800000
	s_add_u32 s24, s21, s18                                    // 000000001640: 80181215
	s_addc_u32 s25, s22, s19                                   // 000000001644: 82191316
	s_add_u32 s18, s18, 0x70                                   // 000000001648: 8012FF12 00000070
	s_load_b512 s[0:15], s[24:25], null                        // 000000001650: F410000C F8000000
	s_addc_u32 s19, s19, 0                                     // 000000001658: 82138013
	s_cmpk_eq_i32 s18, 0x3090                                  // 00000000165C: B1923090
	s_waitcnt lgkmcnt(0)                                       // 000000001660: BF89FC07
	v_add_f32_e32 v0, s0, v0                                   // 000000001664: 06000000
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001668: BF870091
	v_add_f32_e32 v0, s1, v0                                   // 00000000166C: 06000001
	v_add_f32_e32 v0, s2, v0                                   // 000000001670: 06000002
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001674: BF870091
	v_add_f32_e32 v0, s3, v0                                   // 000000001678: 06000003
	v_add_f32_e32 v0, s4, v0                                   // 00000000167C: 06000004
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001680: BF870091
	v_add_f32_e32 v0, s5, v0                                   // 000000001684: 06000005
	v_add_f32_e32 v0, s6, v0                                   // 000000001688: 06000006
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 00000000168C: BF8700A1
	v_add_f32_e32 v0, s7, v0                                   // 000000001690: 06000007
	s_load_b256 s[0:7], s[24:25], 0x40                         // 000000001694: F40C000C F8000040
	v_add_f32_e32 v0, s8, v0                                   // 00000000169C: 06000008
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000016A0: BF870091
	v_add_f32_e32 v0, s9, v0                                   // 0000000016A4: 06000009
	v_add_f32_e32 v0, s10, v0                                  // 0000000016A8: 0600000A
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000016AC: BF870091
	v_add_f32_e32 v0, s11, v0                                  // 0000000016B0: 0600000B
	v_add_f32_e32 v0, s12, v0                                  // 0000000016B4: 0600000C
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000016B8: BF870091
	v_add_f32_e32 v0, s13, v0                                  // 0000000016BC: 0600000D
	v_add_f32_e32 v0, s14, v0                                  // 0000000016C0: 0600000E
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000016C4: BF8700A1
	v_add_f32_e32 v0, s15, v0                                  // 0000000016C8: 0600000F
	s_waitcnt lgkmcnt(0)                                       // 0000000016CC: BF89FC07
	v_add_f32_e32 v0, s0, v0                                   // 0000000016D0: 06000000
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000016D4: BF870091
	v_add_f32_e32 v0, s1, v0                                   // 0000000016D8: 06000001
	v_add_f32_e32 v0, s2, v0                                   // 0000000016DC: 06000002
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000016E0: BF8700A1
	v_add_f32_e32 v0, s3, v0                                   // 0000000016E4: 06000003
	s_load_b128 s[0:3], s[24:25], 0x60                         // 0000000016E8: F408000C F8000060
	v_add_f32_e32 v0, s4, v0                                   // 0000000016F0: 06000004
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000016F4: BF870091
	v_add_f32_e32 v0, s5, v0                                   // 0000000016F8: 06000005
	v_add_f32_e32 v0, s6, v0                                   // 0000000016FC: 06000006
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001700: BF8700A1
	v_add_f32_e32 v0, s7, v0                                   // 000000001704: 06000007
	s_waitcnt lgkmcnt(0)                                       // 000000001708: BF89FC07
	v_add_f32_e32 v0, s0, v0                                   // 00000000170C: 06000000
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001710: BF870091
	v_add_f32_e32 v0, s1, v0                                   // 000000001714: 06000001
	v_add_f32_e32 v0, s2, v0                                   // 000000001718: 06000002
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000171C: BF870001
	v_add_f32_e32 v0, s3, v0                                   // 000000001720: 06000003
	s_cbranch_scc0 65478                                       // 000000001724: BFA1FFC6 <r_64_3108+0x40>
	s_set_inst_prefetch_distance 0x2                           // 000000001728: BF840002
	s_ashr_i32 s21, s20, 31                                    // 00000000172C: 86159F14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001730: BF8704A1
	v_dual_mul_f32 v0, 0x39a8b099, v0 :: v_dual_mov_b32 v1, 0  // 000000001734: C8D000FF 00000080 39A8B099
	s_lshl_b64 s[0:1], s[20:21], 2                             // 000000001740: 84808214
	s_add_u32 s0, s16, s0                                      // 000000001744: 80000010
	s_addc_u32 s1, s17, s1                                     // 000000001748: 82010111
	global_store_b32 v1, v0, s[0:1]                            // 00000000174C: DC6A0000 00000001
	s_nop 0                                                    // 000000001754: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001758: BFB60003
	s_endpgm                                                   // 00000000175C: BFB00000
